"""
MarkerPhish — Resource profiling of the Ollama server
======================================================
Measures what inference costs while a run is in progress:

- process-tree RSS and CPU of the Ollama server (the "serve" process and every
  descendant, e.g. the llama-server runner that holds the loaded model);
- model memory as reported by Ollama's /api/ps (size, size_vram);
- NVIDIA VRAM and GPU utilisation through NVML, when available;
- AC power, so a run that lost mains power can be flagged.

Memory metrics. The primary metric is the model memory from /api/ps: what the
runtime itself allocates for the model (weights + KV cache + compute buffers).
The secondary metric is the process-tree RSS. RSS is noisy: model weights are
memory-mapped, so mapped pages count once they are touched and the OS may drop
or share them (page cache); RSS can therefore be below or above the model's
real footprint, and it also includes the server's own overhead.

The sampler runs in a daemon thread; everything that touches the system is
injectable so the unit tests never read real processes or call Ollama.
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import psutil
import requests

from config import OLLAMA_BASE_URL

from .utils import logger

MB = 1024 * 1024
SAMPLE_INTERVAL_S = 0.5
PS_INTERVAL_S = 2.0
POWER_INTERVAL_S = 5.0
TRAY_APP_NAMES = {"ollama app.exe", "ollama app", "ollama.app"}


# ──────────────────────────────────────────────
# Process matching
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class ProcInfo:
    pid: int
    ppid: int
    name: str
    cmdline: str


def _is_tray(info: ProcInfo) -> bool:
    return info.name.lower() in TRAY_APP_NAMES


def select_server_roots(processes: list[ProcInfo]) -> list[int]:
    """
    PIDs of the Ollama server root(s): a process named ollama* (not the tray app)
    whose command line contains "serve". Fallback when none has "serve" in its
    command line (e.g. access denied): every ollama* process that is not the
    tray app and whose parent is not itself such a process. Names and command
    lines are compared in lowercase, so Windows, macOS and Linux names match.
    """
    ollama = [p for p in processes if p.name.lower().startswith("ollama") and not _is_tray(p)]
    serving = [p.pid for p in ollama if "serve" in p.cmdline.lower()]
    if serving:
        return serving
    pids = {p.pid for p in ollama}
    return [p.pid for p in ollama if p.ppid not in pids]


def descendants(roots: list[int], processes: list[ProcInfo]) -> list[int]:
    """roots plus all their descendants (recursive), never the tray app."""
    children: dict[int, list[ProcInfo]] = {}
    for p in processes:
        children.setdefault(p.ppid, []).append(p)
    tray = {p.pid for p in processes if _is_tray(p)}
    out, stack = [], [r for r in roots if r not in tray]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
        stack.extend(c.pid for c in children.get(pid, []) if c.pid not in tray)
    return out


def list_processes() -> list[ProcInfo]:
    """Snapshot of all processes (name and command line may be empty when access is denied)."""
    procs = []
    for p in psutil.process_iter(["pid", "ppid", "name", "cmdline"]):
        try:
            info = p.info
            procs.append(
                ProcInfo(
                    pid=int(info["pid"]),
                    ppid=int(info.get("ppid") or 0),
                    name=str(info.get("name") or ""),
                    cmdline=" ".join(info.get("cmdline") or []),
                )
            )
        except (psutil.Error, TypeError, ValueError):
            continue
    return procs


def ollama_tree_pids(processes: list[ProcInfo] | None = None) -> list[int]:
    """Full resolution from a process snapshot (slow on Windows: reads every command line)."""
    procs = processes if processes is not None else list_processes()
    return descendants(select_server_roots(procs), procs)


class OllamaTree:
    """
    Cheap per-sample resolution of the Ollama process tree.

    The server root is found once with a full process scan (command lines are
    slow to read on Windows: ~2 s for a full scan). Every later call re-reads only
    the root's descendants (psutil children(recursive=True), a parent-PID
    snapshot), so a runner spawned mid-run is still picked up. The full scan is
    repeated only when the root is gone, at most every `rescan_s` seconds.
    """

    def __init__(self, rescan_s: float = 10.0, clock: Callable[[], float] = time.monotonic):
        self.rescan_s = rescan_s
        self.clock = clock
        self._roots: list[psutil.Process] = []
        self._last_scan = -1e9

    def _scan(self) -> None:
        self._last_scan = self.clock()
        procs = list_processes()
        self._roots = []
        for pid in select_server_roots(procs):
            try:
                self._roots.append(psutil.Process(pid))
            except psutil.Error:
                continue

    def __call__(self) -> list[int]:
        alive = [r for r in self._roots if _running(r)]
        if not alive and self.clock() - self._last_scan >= self.rescan_s:
            self._scan()
            alive = self._roots
        pids: list[int] = []
        for root in alive:
            pids.append(root.pid)
            try:
                pids.extend(
                    c.pid for c in root.children(recursive=True) if c.name().lower() not in TRAY_APP_NAMES
                )
            except psutil.Error:
                continue
        return pids


def runner_cmdlines() -> list[str]:
    """Command lines of the Ollama server's descendants (the model runners); [] if unreadable."""
    pids = OllamaTree()()
    out = []
    for pid in pids[1:]:
        try:
            cmd = " ".join(psutil.Process(pid).cmdline())
        except psutil.Error:
            continue
        if cmd:
            out.append(cmd)
    return out


def _running(proc: psutil.Process) -> bool:
    try:
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


# ──────────────────────────────────────────────
# Ollama API (no model is loaded by these calls)
# ──────────────────────────────────────────────
def ollama_ps(timeout: float = 5) -> list[dict] | None:
    """Loaded models from /api/ps, or None when Ollama cannot be reached."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/ps", timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("models", [])
    except Exception as e:  # noqa: BLE001
        logger.debug("/api/ps unavailable: %s", e)
        return None


def model_matches(entry: dict, model_name: str) -> bool:
    wanted = {model_name} | ({model_name + ":latest"} if ":" not in model_name else set())
    return entry.get("name") in wanted or entry.get("model") in wanted


# ──────────────────────────────────────────────
# NVML (optional)
# ──────────────────────────────────────────────
class _Nvml:
    """Lazy NVML wrapper: None everywhere when the library or a GPU is missing."""

    def __init__(self):
        self.handles = None

    def _init(self) -> bool:
        if self.handles is not None:
            return bool(self.handles)
        try:
            import pynvml

            pynvml.nvmlInit()
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception:  # noqa: BLE001 - no NVIDIA driver / library: silently unavailable
            self.handles = []
        return bool(self.handles)

    def read(self) -> tuple[float | None, float | None]:
        """(VRAM used MB summed over GPUs, mean GPU utilisation %) or (None, None)."""
        if not self._init():
            return None, None
        try:
            import pynvml

            used = sum(pynvml.nvmlDeviceGetMemoryInfo(h).used for h in self.handles) / MB
            util = statistics.mean(pynvml.nvmlDeviceGetUtilizationRates(h).gpu for h in self.handles)
            return round(used, 1), round(float(util), 1)
        except Exception:  # noqa: BLE001
            return None, None


# ──────────────────────────────────────────────
# Process-tree reader
# ──────────────────────────────────────────────
class TreeReader:
    """
    RSS (MB) and CPU (% of the whole machine) of the Ollama process tree.

    The tree is re-resolved on every read, because the runner process appears
    only when a model is loaded and can be replaced mid-run. psutil.Process
    objects are kept between reads so per-process CPU counters are continuous;
    a process seen for the first time contributes 0 % on that read.
    """

    def __init__(self, pid_provider: Callable[[], list[int]] | None = None, logical_cores: int | None = None):
        self.pid_provider = pid_provider or OllamaTree()
        self.cores = logical_cores or psutil.cpu_count(logical=True) or 1
        self._procs: dict[int, psutil.Process] = {}

    def read(self) -> tuple[float, float, int]:
        """(rss_mb, cpu_pct normalised to 0-100 for the whole machine, number of processes)."""
        pids = set(self.pid_provider())
        for pid in list(self._procs):
            if pid not in pids:
                del self._procs[pid]
        rss = 0
        cpu = 0.0
        for pid in pids:
            proc = self._procs.get(pid)
            try:
                if proc is None:
                    proc = psutil.Process(pid)
                    proc.cpu_percent(None)  # prime the counter
                    self._procs[pid] = proc
                    rss += proc.memory_info().rss
                    continue
                rss += proc.memory_info().rss
                cpu += proc.cpu_percent(None)
            except psutil.Error:
                self._procs.pop(pid, None)
        return round(rss / MB, 1), round(min(100.0, cpu / self.cores), 2), len(pids)

    def runner_cmdlines(self, processes: list[ProcInfo] | None = None) -> list[str]:
        """Command lines of the tree's non-root processes (the model runners)."""
        procs = processes if processes is not None else list_processes()
        roots = set(select_server_roots(procs))
        tree = set(descendants(list(roots), procs))
        return [p.cmdline for p in procs if p.pid in tree and p.pid not in roots and p.cmdline]


def read_tree_rss_mb(pid_provider: Callable[[], list[int]] | None = None) -> float | None:
    """One-off RSS of the Ollama process tree in MB (None if no server process is found)."""
    pids = pid_provider() if pid_provider else ollama_tree_pids()
    if not pids:
        return None
    total = 0
    for pid in pids:
        try:
            total += psutil.Process(pid).memory_info().rss
        except psutil.Error:
            continue
    return round(total / MB, 1)


# ──────────────────────────────────────────────
# Sampler
# ──────────────────────────────────────────────
def on_ac_power() -> bool | None:
    try:
        battery = psutil.sensors_battery()
    except Exception:  # noqa: BLE001
        return None
    if battery is None or battery.power_plugged is None:
        return None
    return bool(battery.power_plugged)


@dataclass
class ResourceSampler:
    """
    Background sampler of the Ollama process tree.

    Every `interval_s`: tree RSS, tree CPU %, VRAM used and GPU utilisation.
    About every `ps_interval_s`: /api/ps for the evaluated model (peak size and
    size_vram). About every `power_interval_s`: AC power. The time spent in each
    sampling cycle is measured (sampler_cost_ms) so the profiler's own overhead
    is reported. start() takes the first sample synchronously (the baseline).
    """

    model_name: str
    interval_s: float = SAMPLE_INTERVAL_S
    ps_interval_s: float = PS_INTERVAL_S
    power_interval_s: float = POWER_INTERVAL_S
    tree: TreeReader = field(default_factory=TreeReader)
    ps_fetcher: Callable[[], list[dict] | None] = ollama_ps
    gpu_reader: Callable[[], tuple[float | None, float | None]] = field(default_factory=lambda: _Nvml().read)
    power_reader: Callable[[], bool | None] = on_ac_power
    clock: Callable[[], float] = time.perf_counter

    samples: list[dict] = field(default_factory=list)
    _gpu_util: list[float] = field(default_factory=list)
    _costs_ms: list[float] = field(default_factory=list)
    _ps_peak_size: int | None = None
    _ps_peak_vram: int | None = None
    _ps_polls: int = 0
    _power: list[bool | None] = field(default_factory=list)
    _max_procs: int = 0

    def __post_init__(self):
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self._last_ps = -1e9
        self._last_power = -1e9

    # ── one cycle ──
    def _sample(self) -> None:
        start = self.clock()
        t = start - self._t0
        rss, cpu, n = self.tree.read()
        self._max_procs = max(self._max_procs, n)
        vram, util = self.gpu_reader()
        if util is not None:
            self._gpu_util.append(util)
        self.samples.append({"t_s": round(t, 3), "ram_mb": rss, "cpu_pct": cpu, "vram_mb": vram})
        if t - self._last_ps >= self.ps_interval_s:
            self._last_ps = t
            self._poll_ps()
        if t - self._last_power >= self.power_interval_s:
            self._last_power = t
            self._power.append(self.power_reader())
        self._costs_ms.append((self.clock() - start) * 1000)

    def _poll_ps(self) -> None:
        models = self.ps_fetcher()
        if models is None:
            return
        self._ps_polls += 1
        for entry in models:
            if model_matches(entry, self.model_name):
                size = int(entry.get("size") or 0)
                vram = int(entry.get("size_vram") or 0)
                self._ps_peak_size = max(self._ps_peak_size or 0, size)
                self._ps_peak_vram = max(self._ps_peak_vram or 0, vram)

    def _loop(self) -> None:
        next_t = self.clock() + self.interval_s
        while not self._stop.wait(max(0.0, next_t - self.clock())):
            try:
                self._sample()
            except Exception as e:  # noqa: BLE001 - a sampler error must never break the run
                logger.debug("Resource sample failed: %s", e)
            next_t += self.interval_s

    # ── lifecycle ──
    def start(self) -> float:
        """Take the baseline sample and start the thread. Returns the baseline RSS (MB)."""
        self._t0 = self.clock()
        self._sample()
        self._thread = threading.Thread(target=self._loop, name="markerphish-sampler", daemon=True)
        self._thread.start()
        return self.samples[0]["ram_mb"]

    def stop(self) -> dict:
        """Take a final sample, stop the thread and return the summary."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        try:
            self._last_ps = -1e9  # final /api/ps poll
            self._sample()
        except Exception as e:  # noqa: BLE001
            logger.debug("Final resource sample failed: %s", e)
        return self.summary()

    def summary(self) -> dict:
        ram = [s["ram_mb"] for s in self.samples]
        cpu = [s["cpu_pct"] for s in self.samples[1:]]  # the baseline's CPU counter is unprimed
        vram = [s["vram_mb"] for s in self.samples if s["vram_mb"] is not None]
        power = [p for p in self._power if p is not None]
        return {
            "peak_ram_mb": max(ram) if ram else None,
            "mean_ram_mb": round(statistics.mean(ram), 1) if ram else None,
            "baseline_ram_mb": ram[0] if ram else None,
            "peak_cpu_pct": max(cpu) if cpu else None,
            "mean_cpu_pct": round(statistics.mean(cpu), 2) if cpu else None,
            "model_size_mb": round(self._ps_peak_size / MB, 1) if self._ps_peak_size else None,
            "model_vram_mb": round(self._ps_peak_vram / MB, 1) if self._ps_peak_vram is not None else None,
            "peak_vram_mb": max(vram) if vram else None,
            "mean_vram_mb": round(statistics.mean(vram), 1) if vram else None,
            "mean_gpu_util_pct": round(statistics.mean(self._gpu_util), 1) if self._gpu_util else None,
            "sample_interval_s": self.interval_s,
            "n_samples": len(self.samples),
            "ps_polls": self._ps_polls,
            "sampler_cost_ms": round(statistics.mean(self._costs_ms), 3) if self._costs_ms else None,
            "sampler_cost_max_ms": round(max(self._costs_ms), 3) if self._costs_ms else None,
            "max_tree_processes": self._max_procs,
            "ac_power_lost_during_run": (False in power) if power else None,
            "samples": list(self.samples),
        }
