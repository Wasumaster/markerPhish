"""backend.profiler, backend.profiles and the warm / cold run lifecycle."""

import time

import pytest

from backend import evaluator, profiler
from backend.profiler import ProcInfo, ResourceSampler, descendants, select_server_roots
from backend.profiles import fits_profiles, profile_budget_mb
from backend.run_report import build_run_report
from backend.sampling import balanced_sample

WIN = [
    ProcInfo(1, 0, "explorer.exe", ""),
    ProcInfo(10, 1, "ollama app.exe", r"C:\Ollama\ollama app.exe"),
    ProcInfo(11, 10, "ollama.exe", r"C:\Ollama\ollama.exe serve"),
    ProcInfo(12, 11, "llama-server.exe", "llama-server.exe --model blob --threads 8"),
    ProcInfo(13, 12, "conhost.exe", ""),
    ProcInfo(20, 1, "python.exe", "streamlit run app.py"),
]


# ── process matching ──
def test_server_root_is_the_serve_process_not_the_tray_app():
    assert select_server_roots(WIN) == [11]


def test_tree_includes_descendants_recursively_but_not_the_tray_app():
    assert sorted(descendants([11], WIN)) == [11, 12, 13]
    assert descendants([10], WIN) == []  # the tray app is never part of the tree


def test_linux_and_macos_names_match_case_insensitively():
    procs = [ProcInfo(5, 1, "Ollama", "/usr/local/bin/ollama SERVE"), ProcInfo(6, 5, "ollama_llama_server", "--model x")]
    assert select_server_roots(procs) == [5]
    assert sorted(descendants([5], procs)) == [5, 6]


def test_fallback_without_serve_in_any_command_line():
    procs = [ProcInfo(10, 1, "ollama app.exe", ""), ProcInfo(11, 10, "ollama.exe", ""), ProcInfo(12, 11, "ollama.exe", "")]
    assert select_server_roots(procs) == [11]  # 12's parent is itself an ollama process


def test_no_ollama_process():
    assert select_server_roots([ProcInfo(1, 0, "init", "")]) == []
    assert profiler.ollama_tree_pids([]) == []


def test_runner_command_lines():
    assert profiler.TreeReader(lambda: []).runner_cmdlines(WIN) == [
        "llama-server.exe --model blob --threads 8"
    ]


# ── tree reader (real psutil on the test process itself) ──
def test_tree_reader_reads_rss_and_normalises_cpu():
    import os

    reader = profiler.TreeReader(lambda: [os.getpid()], logical_cores=16)
    rss, cpu, n = reader.read()  # first read primes the CPU counter
    assert rss > 10 and cpu == 0 and n == 1
    rss2, cpu2, _ = reader.read()
    assert rss2 > 10 and 0 <= cpu2 <= 100


def test_tree_cpu_is_normalised_to_the_whole_machine(monkeypatch):
    class FakeProc:
        def __init__(self, pid):
            self.pid = pid

        def cpu_percent(self, interval):
            return 400.0  # psutil: 4 cores fully busy

        def memory_info(self):
            class M:
                rss = 100 * 1024 * 1024
            return M()

    monkeypatch.setattr(profiler.psutil, "Process", FakeProc)
    reader = profiler.TreeReader(lambda: [1, 2], logical_cores=16)
    reader.read()  # primes
    rss, cpu, n = reader.read()
    assert (rss, n) == (200.0, 2)
    assert cpu == 50.0  # (400 + 400) / 16 cores


def test_tree_reader_drops_vanished_processes():
    reader = profiler.TreeReader(lambda: [999_999_999], logical_cores=4)
    assert reader.read() == (0.0, 0.0, 1)


# ── sampler ──
class FakeTree:
    def __init__(self, values):
        self.values = list(values)

    def read(self):
        v = self.values.pop(0) if len(self.values) > 1 else self.values[0]
        return v, 40.0, 2


def _sampler(**kw):
    defaults = dict(
        model_name="m:1",
        interval_s=0.02,
        ps_interval_s=0.0,
        power_interval_s=0.0,
        tree=FakeTree([500.0, 900.0, 2500.0, 2400.0]),
        ps_fetcher=lambda: [{"name": "m:1", "size": 3 * 1024**3, "size_vram": 0}, {"name": "other", "size": 9}],
        gpu_reader=lambda: (None, None),
        power_reader=lambda: True,
    )
    defaults.update(kw)
    return ResourceSampler(**defaults)


def test_sampler_summary():
    s = _sampler()
    assert s.start() == 500.0  # baseline = first, synchronous sample
    time.sleep(0.15)
    out = s.stop()
    assert out["baseline_ram_mb"] == 500.0
    assert out["peak_ram_mb"] == 2500.0
    assert out["n_samples"] >= 4 and len(out["samples"]) == out["n_samples"]
    assert out["model_size_mb"] == 3072.0 and out["model_vram_mb"] == 0.0
    assert out["peak_vram_mb"] is None and out["mean_gpu_util_pct"] is None
    assert out["mean_cpu_pct"] == 40.0
    assert out["sampler_cost_ms"] is not None and out["sampler_cost_ms"] >= 0
    assert out["ac_power_lost_during_run"] is False
    assert set(out["samples"][0]) == {"t_s", "ram_mb", "cpu_pct", "vram_mb"}


def test_sampler_records_gpu_and_power_loss():
    power = iter([True, True, False])
    s = _sampler(gpu_reader=lambda: (1234.0, 55.0), power_reader=lambda: next(power, False))
    s.start()
    time.sleep(0.1)
    out = s.stop()
    assert out["peak_vram_mb"] == 1234.0 and out["mean_gpu_util_pct"] == 55.0
    assert out["ac_power_lost_during_run"] is True


def test_sampler_survives_errors_and_missing_ollama():
    class Broken:
        def read(self):
            raise RuntimeError("psutil exploded")

    s = _sampler(ps_fetcher=lambda: None)
    s.start()
    s.tree = Broken()
    time.sleep(0.06)
    out = s.stop()
    assert out["model_size_mb"] is None
    assert out["n_samples"] >= 1


def test_nvml_never_raises(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_pynvml(name, *a, **k):
        if name == "pynvml":
            raise ImportError("not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_pynvml)
    assert profiler._Nvml().read() == (None, None)


# ── constraint profiles ──
PROFILES = {
    "laptop-8gb": {"ram_gb": 8, "gpu": False, "max_p95_latency_s": None},
    "laptop-16gb": {"ram_gb": 16, "gpu": False, "max_p95_latency_s": 5.0},
}


def test_profile_budget():
    assert profile_budget_mb({"ram_gb": 8}, 0.5) == 4096.0


def test_fits_profiles_uses_model_memory_and_optional_latency():
    res = {"model_size_mb": 5000.0, "peak_ram_mb": 6100.0, "model_vram_mb": 0}
    out = fits_profiles(res, p95_latency_s=4.0, profiles=PROFILES, headroom=0.5)
    assert out["laptop-8gb"]["fits_profile"] is False and out["laptop-8gb"]["fits_memory"] is False
    assert out["laptop-16gb"]["fits_profile"] is True
    assert out["laptop-16gb"]["rss_within_budget"] is True
    slow = fits_profiles(res, p95_latency_s=9.0, profiles=PROFILES, headroom=0.5)
    assert slow["laptop-16gb"]["fits_latency"] is False and slow["laptop-16gb"]["fits_profile"] is False


def test_fits_profiles_unknown_memory_and_gpu_offload_flag():
    out = fits_profiles({"model_size_mb": None}, 1.0, profiles=PROFILES)
    assert out["laptop-8gb"]["fits_profile"] is None
    gpu = fits_profiles({"model_size_mb": 1000.0, "model_vram_mb": 800.0}, 1.0, profiles=PROFILES)
    assert gpu["laptop-8gb"]["measured_with_gpu_offload"] is True


# ── run lifecycle with the fake Ollama ──
@pytest.fixture
def sample(toy_df):
    return balanced_sample(toy_df.assign(source_id=[f"t:{i}" for i in toy_df["id"]]), 20).df


def _quick_sampler(model_name):
    return ResourceSampler(
        model_name,
        interval_s=0.01,
        tree=FakeTree([300.0, 1500.0]),
        gpu_reader=lambda: (None, None),
        power_reader=lambda: True,
    )


def test_warm_start(fake_ollama, sample):
    run = evaluator.run_evaluation(
        "qwen2.5:1.5b", sample, num_gpu=0, prompt_variant="zeroshot",
        sampler_factory=_quick_sampler, tree_rss_reader=lambda: 42.0,
    )
    assert run.start_mode == "warm"
    assert run.cold_start_s == pytest.approx(2.0)  # paid by the excluded warm-up request
    assert not any(r.is_cold_request for r in run.results)
    assert run.pre_warmup_ram_mb == 42.0 and run.baseline_ram_mb == 300.0
    assert run.resources["peak_ram_mb"] == 1500.0
    assert run.resources["model_size_mb"] is not None  # /api/ps listed the loaded model
    assert len(fake_ollama.generate_calls()) == 21  # warm-up + 20


def test_cold_start_unloads_and_excludes_the_first_latency(fake_ollama, sample):
    fake_ollama.load("qwen2.5:1.5b")  # already loaded: cold must unload it first
    run = evaluator.run_evaluation(
        "qwen2.5:1.5b", sample, num_gpu=0, prompt_variant="zeroshot", start_mode="cold",
        sampler_factory=_quick_sampler, tree_rss_reader=lambda: 42.0,
    )
    assert run.cold_unload_confirmed is True
    assert run.results[0].is_cold_request and not any(r.is_cold_request for r in run.results[1:])
    assert run.cold_start_s == pytest.approx(2.0)  # load_duration of the first request
    assert len(fake_ollama.generate_calls()) == 21  # unload + 20 (no warm-up)

    report = build_run_report(run, df_sample=sample, dataset_choice="x", sample_info=None,
                              code={"commit": None, "dirty": None, "changed_files": None})
    assert report["resource_metrics"]["latency_excluded_cold_requests"] == 1
    assert report["raw_predictions"][0]["is_cold_request"] is True
    assert report["classification_metrics"]["total_evaluated"] == 20  # its prediction still counts
    assert report["resources"]["start_mode"] == "cold"


def test_cold_start_timeout_is_recorded(fake_ollama, sample, monkeypatch):
    fake_ollama.load("qwen2.5:1.5b")
    fake_ollama.unload_ignored.add("qwen2.5:1.5b")  # the server never unloads it
    monkeypatch.setattr(evaluator, "COLD_UNLOAD_TIMEOUT_S", 0.05)
    run = evaluator.run_evaluation(
        "qwen2.5:1.5b", sample, num_gpu=0, prompt_variant="zeroshot", start_mode="cold",
        sampler_factory=_quick_sampler, tree_rss_reader=lambda: 1.0,
    )
    assert run.cold_unload_confirmed is False


def test_other_loaded_models_are_reported_and_unloaded(fake_ollama, sample):
    fake_ollama.load("llama3.2:latest")
    run = evaluator.run_evaluation(
        "qwen2.5:1.5b", sample, num_gpu=0, prompt_variant="zeroshot",
        sampler_factory=_quick_sampler, tree_rss_reader=lambda: 1.0,
    )
    assert run.prerun["ollama_loaded_models"] == ["llama3.2:latest"]
    assert run.prerun["other_models_loaded"] is True
    assert run.prerun["other_models_unloaded"] == ["llama3.2:latest"]
    assert "llama3.2:latest" not in fake_ollama.loaded


def test_other_models_can_be_kept(fake_ollama, sample):
    fake_ollama.load("llama3.2:latest")
    run = evaluator.run_evaluation(
        "qwen2.5:1.5b", sample, num_gpu=0, prompt_variant="zeroshot", unload_other_models=False,
        sampler_factory=_quick_sampler, tree_rss_reader=lambda: 1.0,
    )
    assert run.prerun["other_models_loaded"] is True and run.prerun["other_models_unloaded"] == []
    assert "llama3.2:latest" in fake_ollama.loaded


def test_unknown_start_mode_is_rejected(fake_ollama, sample):
    with pytest.raises(ValueError, match="start mode"):
        evaluator.run_evaluation("qwen2.5:1.5b", sample, start_mode="lukewarm")


@pytest.mark.parametrize(
    "readings, expected",
    [([True, True], (False, False)), ([True, False], (True, True)), ([False, False], (False, True)),
     ([None, True, None], (False, False)), ([], (False, False))],
)
def test_power_changed(readings, expected):
    assert evaluator.power_changed(readings) == expected


def test_report_has_resources_hardware_and_power(fake_ollama, sample, monkeypatch):
    from backend import manifest

    states = iter([{"on_ac_power": True}, {"on_ac_power": False}])
    monkeypatch.setattr(evaluator, "power_state", lambda: next(states))
    run = evaluator.run_evaluation(
        "qwen2.5:1.5b", sample, num_gpu=0, prompt_variant="zeroshot",
        sampler_factory=_quick_sampler, tree_rss_reader=lambda: 1.0,
    )
    report = build_run_report(run, df_sample=sample, dataset_choice="x", sample_info=None,
                              code={"commit": None, "dirty": None, "changed_files": None})
    power = report["manifest"]["power"]
    assert power["power_changed"] is True and power["on_battery_during_run"] is True
    res = report["resources"]
    for key in ("peak_ram_mb", "mean_ram_mb", "baseline_ram_mb", "pre_warmup_ram_mb", "peak_cpu_pct", "mean_cpu_pct",
                "model_size_mb", "model_vram_mb", "peak_vram_mb", "mean_vram_mb", "mean_gpu_util_pct",
                "sample_interval_s", "n_samples", "sampler_cost_ms", "start_mode", "ollama_loaded_models",
                "other_models_loaded", "fits_profile", "samples"):
        assert key in res, key
    assert set(res["fits_profile"]) == {"laptop-8gb", "laptop-16gb"}
    hw = report["hardware"]
    for key in ("cpu_name", "cpu_cores_physical", "cpu_cores_logical", "ram_total_gb", "gpu_name",
                "gpu_vram_mb", "os_version", "python_version", "ollama_version"):
        assert key in hw, key
    assert report["hardware_profile"] == hw  # old key kept, now the per-run scan
    assert manifest.report_protocol_version(report) == 2


# ── cached tree resolution ──
def test_ollama_tree_scans_once_then_follows_children(monkeypatch):
    import os

    me = os.getpid()
    scans = []

    def fake_scan():
        scans.append(1)
        return [ProcInfo(me, 1, "ollama.exe", "ollama.exe serve")]

    monkeypatch.setattr(profiler, "list_processes", fake_scan)
    clock = iter(range(0, 1000)).__next__
    tree = profiler.OllamaTree(rescan_s=10.0, clock=lambda: float(clock()))
    assert tree()[0] == me
    assert tree()[0] == me
    assert len(scans) == 1  # the root is alive: no second full scan


def test_ollama_tree_rescans_at_most_every_rescan_s(monkeypatch):
    scans = []
    monkeypatch.setattr(profiler, "list_processes", lambda: scans.append(1) or [])
    now = [0.0]
    tree = profiler.OllamaTree(rescan_s=10.0, clock=lambda: now[0])
    assert tree() == [] and len(scans) == 1
    now[0] = 5.0
    assert tree() == [] and len(scans) == 1  # throttled: no server, no repeated slow scans
    now[0] = 11.0
    tree()
    assert len(scans) == 2


def test_ollama_url_avoids_localhost():
    import config

    assert "localhost" not in config.OLLAMA_BASE_URL  # IPv6 fallback adds ~2 s per request on Windows
