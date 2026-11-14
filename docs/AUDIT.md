<!-- Recovered verbatim from the audit session transcript (2026-11-13); docs/AUDIT.md did not exist in the repo. Original language: Polish. -->

# Audyt MarkerPhish względem tematu pracy inżynierskiej

Audyt był wyłącznie do odczytu: nie zmieniłem żadnego pliku projektu ani wyników i nie uruchamiałem nowych ewaluacji. Skrypty pomocnicze (profil danych, bootstrap CI) zapisałem w katalogu tymczasowym poza repozytorium. Plan profilowania zasobów czeka wstrzymany.

## 1. Podsumowanie

Aplikacja dobrze pokrywa „rdzeń inżynierski” tematu. Integruje dwa publiczne zbiory, uruchamia lokalne modele przez Ollamę z deterministycznymi ustawieniami i poprawnie liczy precision/recall/F1 dla klasy „malicious”. Mierzy też latencję i przepustowość, zapisując wszystko do JSON. Najsłabsza jest część badawcza obiecana w opisie:
- **pamięć:** zużycie pamięci (*memory footprint*) w ogóle nie jest mierzone;
- **optymalny wybór:** nic nie wyznacza „optymalnego rozwiązania”, platforma tylko pokazuje liczby obok siebie;
- **powtarzalność:** w raporcie brakuje kluczowych metadanych (zbiór danych, ID próbek, digest/kwantyzacja modelu, wersja Ollamy i kodu).

Poważny problem metodologiczny jest w danych:
- **ucięte sygnały:** model nie widzi pól `URL` i `Sender` z PhishFuzzer, a 31–61% e-maili jest obcinanych do 1 500 znaków, więc w większości e-maili brakuje dokładnie tych sygnałów, o które prosi prompt;
- **duplikaty SMS:** zbiór SMS zawiera ~11% duplikatów.

Próbki są małe: przy n=100 przedział ufności F1 ma szerokość ~0,12, a przy n=50 ~0,24–0,27. Bez powtórzeń i przedziałów ufności różnice między modelami będą nierozstrzygalne. Ponadto wszystkie zapisane wyniki w `results/` to runy „legacy” (stary trzyklasowy prompt, `num_ctx` 2048, bez warm-upu). Wyniki do pracy trzeba więc wygenerować od nowa, po poprawkach.

## 2. Tabela zgodności

| Wymaganie | Status | Gdzie w kodzie | Komentarz |
|---|---|---|---|
| Przegląd istniejących metod LLM | poza kodem | — | Projekt (zero-/few-shot/CoT, binarna klasyfikacja) da się odnieść do literatury, ale brakuje typowych punktów odniesienia (pkt 4 i 10 poniżej). |
| Platforma do automatycznej ewaluacji | częściowo | `frontend/runner.py`, `backend/evaluator.py:run_evaluation` | Jeden run = jeden model × wariant × kliknięcie. Brak trybu wsadowego (macierz modeli × wariantów), brak CLI, przebieg blokuje UI. |
| Powtarzalna ewaluacja | częściowo | `config.py` (`OLLAMA_TEMPERATURE=0.0`, `SAMPLE_SEED`), `backend/data_loader.py:stratified_sample` | Wyniki są de facto deterministyczne (dwa runy phi3.5: 50/50 identycznych odpowiedzi). JSON nie zawiera jednak zbioru danych, ID próbek źródłowych, digestu/kwantyzacji modelu, wersji Ollamy, commita ani tekstu/hasha promptu. Brak `seed` w opcjach Ollamy. |
| Analiza i preprocessing publicznych zbiorów | częściowo | `backend/data_loader.py`, `backend/utils.py:clean_text` | Mapowanie etykiet poprawne. Brak deduplikacji (SMS: 778 dokładnych duplikatów). Pola `URL`/`Sender` są pomijane, a obcięcie do 1 500 znaków usuwa linki. Spam odrzucony bez uzasadnienia w dokumentacji. |
| Integracja zbiorów z platformą | spełnione | `DATASET_CHOICES`, `load_unified_dataset` | 6 kombinacji, ujednolicony schemat `id, text, true_label, source`. |
| Kontrolowane eksperymenty | częściowo | `warm_up_model`, `stratified_sample`, `PROMPT_VARIANTS` | Nowe runy: ta sama próbka, warm-up, stałe parametry. Bez kontroli: inne modele załadowane w Ollamie, obciążenie tła, stan cache. Runy legacy są nieporównywalne, a mimo to domyślnie trafiają do Compare. |
| Modele różnej wielkości | częściowo | `config.DEFAULT_MODELS`, `resource_metrics.model_size_mb` | Zainstalowane 1.5B–8B (qwen2.5, llama3.2, phi3.5, gemma3, llama3.1). Zapisywany jest tylko rozmiar na dysku. Brak liczby parametrów i kwantyzacji w JSON, choć `/api/show` je zwraca. |
| Precision, recall, F1 | spełnione (z zastrzeżeniem) | `backend/metrics.py:compute_classification_metrics` | Wzory poprawne, klasa pozytywna = malicious. Odpowiedzi „unknown” są wyłączane z mianownika, co zawyża metryki modeli, które „odmawiają” na trudnych przypadkach. Liczba unknown jest raportowana osobno. |
| Latencja inferencji | spełnione | `compute_resource_metrics` | Średnia, mediana, P95, min, max. Warm-up wykluczony (tylko nowe runy). Przepustowość liczona z czasu, który obejmuje renderowanie UI. Liczniki tokenów są zbierane, ale nie zapisywane. |
| Memory footprint | brak | — | Mierzony jest tylko rozmiar pliku modelu z `/api/tags`. Pokrywa to wstrzymany plan profilowania. |
| Wymagania sprzętowe | częściowo | `backend/hardware.py:scan_hardware`, `hardware_profile` w JSON | Specyfikacja hosta jest zapisywana (tylko w nowszych runach; 1 z 6 runów legacy). Brak powiązania „model X wymaga Y GB RAM”. |
| Analiza kompromisów rozmiar / skuteczność / koszt | częściowo | `frontend/multi_comparison.py` | Wykresy F1 vs P95 i przepustowość vs rozmiar, pogrubienie najlepszej wartości w kolumnie. Brak frontu Pareto i rankingu z ograniczeniami. |
| Wskazanie optymalnego rozwiązania | brak | — | Nic nie wyznacza optimum: ani reguły decyzyjnej, ani metryki kosztowej (np. F1 przy RAM ≤ X i P95 ≤ Y). |
| Środowiska o ograniczonych zasobach | częściowo | `DEFAULT_NUM_GPU=0`, `resolve_num_gpu`, małe modele | Tryb CPU i małe modele są. Brak limitów pamięci/wątków (`num_thread`), porównania kwantyzacji (Q4 vs Q8) i pomiaru RAM. Tytuł PL mówi o „optymalizacji zasobów”, a platforma niczego nie optymalizuje, tylko mierzy. |

## 3. Mocne strony

- **Poprawne metryki binarne:** `backend/metrics.py` liczy TP/TN/FP/FN jawnie, z pozytywem = malicious. Specificity, FNR i FPR mają poprawne wzory, macierz pomyłek ma porządek `[[TP, FN], [FP, TN]]`. Metryki kanałowe są zachowane osobno (`channel_metrics`).
- **Kanał spoza modelu:** w nowych runach kanał wynika ze źródła danych (`config.channel_for_source`), a nie z odpowiedzi modelu. Usuwa to artefakt, w którym model „mylił kanał” mimo poprawnego wykrycia zagrożenia.
- **Wymuszony format odpowiedzi:** JSON Schema w parametrze `format` (`backend/evaluator.py:build_output_schema`) plus parser z fallbackami (`parse_binary_label`) w praktyce eliminują odpowiedzi nieparsowalne. Smoke test z poprzedniej sesji: 0 unknown we wszystkich trzech wariantach.
- **Kontrolowany eksperyment promptowy:** trzy warianty (`config.PROMPT_VARIANTS`: zeroshot / fewshot / fewshotcot) różnią się jedną zmienną naraz, a przykłady few-shot są pisane ręcznie, nie pochodzą ze zbiorów testowych, więc nie przeciekają do ewaluacji.
- **Odtwarzalna próbka:** `stratified_sample` (warstwy źródło × etykieta, metoda największych reszt, stały seed), więc każdy model i wariant widzi te same rekordy.
- **Determinizm potwierdzony empirycznie:** dwa runy phi3.5 na tej samej próbce dały identyczne odpowiedzi w 50/50 przypadków. Latencja różniła się o medianę 5,2% (maks. 15%), co jest dobrym argumentem do rozdziału metodologicznego.
- **Odporność na awarie GPU:** fallback na CPU (`_post_generate`) i tryb CPU Safe Mode są udokumentowane w OVERVIEW §7.
- **Wsteczna kompatybilność raportów:** `ensure_report_compatibility` przelicza metryki binarne dla starych plików, więc stare runy nadal się wczytują.

## 4. Słabe strony i ryzyka

### Krytyczne

1. **Model nie widzi kluczowych sygnałów phishingu** (`backend/data_loader.py:_load_phishfuzzer_json`, linie z `subject`/`body`):
   - **pominięte pola:** loader bierze tylko `Subject` + `Body` i pomija pola `Sender` oraz `URL`;
   - **brak linków:** w seedzie 2 723 z 3 300 e-maili ma wypełnione `URL`, a tylko 978 ma ten URL także w treści;
   - **obcinanie:** `clean_text` tnie do 1 500 znaków (31,4% e-maili seed i 60,5% rephrased); w 155 e-mailach seed pierwszy link pojawia się dopiero po 1 500. znaku.

   Prompt każe oceniać domeny linków i podszywanie się pod nadawcę, ale w większości e-maili tych danych nie ma. Recenzent zapyta, czy mierzymy zdolność modelu, czy artefakt preprocessingu. Wyniki dla e-maili są zaniżone w sposób zależny od zbioru.
2. **Brak pomiaru pamięci** mimo że opis tematu wprost wymienia *memory footprint*. Pokrywa to wstrzymany plan.
3. **Brak wyznaczania „optymalnego rozwiązania”:** `frontend/multi_comparison.py` tylko pogrubia najlepszą wartość w każdej kolumnie osobno. Brak frontu Pareto (F1 vs RAM/latencja), rankingu przy ograniczeniach i jednej funkcji celu. Opis pracy obiecuje identyfikację optimum.
4. **Rozmiar próby i brak niepewności:** przedziały bootstrap 95% dla F1 z zapisanych runów:
   - llama3.2, n=100: F1 0,893 [0,825; 0,946], szerokość 0,12;
   - llama3.2, n=50: 0,844 [0,706; 0,943], szerokość 0,24;
   - phi3.5, n=50: 0,741 [0,591; 0,857], szerokość 0,27.

   Przedziały llama3.2 i phi3.5 przy n=50 się nakładają. Platforma nie liczy CI, nie wspiera powtórzeń ani testów istotności (np. McNemar na parach odpowiedzi), więc wnioski „model A lepszy od B” mogą być nieuprawnione.
5. **Raport nie wystarcza do powtórzenia runu.** W JSON brakuje:
   - wybranego zbioru (`dataset_choice`) i jawnego seeda próbki;
   - mapowania ID próbki na rekord źródłowy (ID są nadawane od nowa 1..n, a numer `No.` / `Original_ID` z PhishFuzzer i numer wiersza CSV giną);
   - digestu, kwantyzacji i liczby parametrów modelu;
   - wersji Ollamy;
   - commita kodu (repo ma obecnie dużo niezacommitowanych zmian);
   - tekstu lub hasha promptu (jest tylko nazwa wariantu, a tekst może się zmienić w kodzie).
6. **Wszystkie zapisane wyniki są legacy i nieporównywalne z obecnym potokiem:** stary trzyklasowy prompt, `num_ctx` 2048, brak warm-upu (pierwsza latencja 22–38 s przy medianie 6–12 s), niestratyfikowana próbka i inny podział klas (np. 49/35/16 zamiast 50/50). Mimo to Compare zaznacza je domyślnie razem z nowymi. Run `eval_qwen2.5_1.5b_20261017_131001.json` ma 100/100 unknown, a mimo to w Compare dostaje pogrubione „najlepsze” false alarm = 0,00%.

### Ważne

7. **Duplikaty i pochodzenie danych:**
   - **duplikaty SMS:** zbiór SMS (Mendeley) ma 778 dokładnych i 1 039 znormalizowanych duplikatów (11–15%); `load_smishing_csv` nie deduplikuje;
   - **powiązane rekordy:** kombinacja „all” łączy seed z jego sześcioma przeredagowaniami (`Original_ID`), więc rekordy nie są niezależne; nie chodzi o wyciek do treningu (model nie jest trenowany), tylko o zawyżoną efektywną wielkość próby;
   - **dane syntetyczne:** zbiór rephrased jest generowany przez LLM (`Created by: LLM`, Gemini), co może faworyzować modele LLM; trzeba to opisać w metodologii.
8. **Zbalansowane klasy a realne proporcje:** próbki są 50/50, a w realnym ruchu phishing to ułamek procenta. Precision z próby zbalansowanej jest silnie zawyżona względem wdrożenia. Platforma nie raportuje precision przy zadanym odsetku zagrożeń w ruchu (*base rate*) ani wag kosztów FP/FN.
9. **Brak metryk binarnych per kanał:** `channel_metrics` liczy klasy phishing/smishing/valid, gdzie „valid” łączy e-maile i SMS-y. Nie da się więc odczytać np. FPR na prawidłowych e-mailach oraz osobno na prawidłowych SMS-ach. Nowe `raw_predictions` mają pole `channel`, ale nikt z niego nie korzysta.
10. **Pomijanie unknown zniekształca metryki:** odpowiedzi „unknown” są wyłączane z precision/recall/F1. Model, który odmawia na trudnych przypadkach, wypada lepiej. Brakuje wariantu „unknown = błąd” (np. traktowane jak safe, czyli przepuszczone zagrożenie). Liczba unknown jest raportowana, ale nie wchodzi do F1.
11. **Pomiar czasu:**
    - **przepustowość:** `throughput_msgs_per_min` liczy z `elapsed_time`, który obejmuje renderowanie tabel Streamlit w `progress_callback` (`frontend/runner.py`), więc mierzy też narzut UI;
    - **nieudane żądania:** timeouty (do 300 s) i błędy wchodzą do statystyk latencji;
    - **tokeny:** `prompt_eval_count` i `eval_count` (oraz czasy `prompt_eval_duration`/`eval_duration`) są zbierane, ale nie trafiają do JSON, więc nie da się policzyć tokenów/s ani rozdzielić prefill od generacji, choć to istotne przy porównaniu wariantów promptu.
12. **Brak automatyzacji serii:** pełny eksperyment (5 modeli × 3 warianty × powtórzenia) wymaga 15+ ręcznych kliknięć, a każdy run blokuje sesję.
13. **Brak kontroli stanu Ollamy:** keep_alive domyślnie 5 min, więc kilka modeli może być załadowanych naraz, co zmienia latencję i pamięć; nic tego nie sprawdza ani nie zapisuje. Częściowo pokrywa to wstrzymany plan (warm/cold, lista załadowanych modeli).
14. **Brak testów w repozytorium:** `git ls-files` nie zawiera żadnych testów. Testy z poprzednich sesji były jednorazowe i leżą poza repo.
15. **Rozjazd dokumentacji z kodem:**
    - `app.py:5`: „phishing and smishing detection with concurrent hardware resource profiling”. Nie ma profilowania współbieżnego.
    - `README.md:74`: „The platform detects the host CPU, RAM, and GPU in real time”. Skan jest jednorazowy, przy starcie sesji (`app.py:init_session`).
    - `README.md:101`: „metrics.py # Binary threat metrics & resource profiling”. Jest tylko statystyka latencji.
    - `README.md:48`: „makes experiments fully reproducible, since the same model version can be re-run”. Wersja (digest) modelu nie jest zapisywana.
    - `OVERVIEW.md:35`: „model size in memory”. To rozmiar pliku na dysku z `/api/tags`.
    - `OVERVIEW.md:239–240` i JSON w §8 (`precision_weighted`, `recall_weighted`, `f1_weighted`). Kod tych metryk nie liczy.
    - `OVERVIEW.md` §9: „F1-Score (Macro) … the main metric for comparing models”. Kod i UI używają binarnego F1 klasy malicious.
    - `OVERVIEW.md:305` (wykresy kołowe) i `README.md:129, 158` („KPI”, „Executive KPI Cards”). Po redesignie tego już nie ma.
16. **Przypinanie zależności:** `requirements.txt` ma wyłącznie `>=`, bez lock-file. `streamlit>=1.38.0`, a kod używa API z 1.66 (`width="stretch"`, `type="tertiary"`, `vertical_alignment`) i selektorów DOM z 1.66, więc starsza wersja się wywróci. `matplotlib` i `seaborn` są w wymaganiach, ale nigdzie nie są importowane.

### Drobne

17. **Bug „select all”:** `frontend/multi_comparison.py:187` zapisuje `multi_selected_files`, którego multiselect nie czyta, więc przycisk nic nie robi.
18. **Bug `is_running`:** w `frontend/runner.py` brak `try/finally`. Wyjątek w trakcie runu albo przerwanie skryptu przez interakcję użytkownika (rerun Streamlita) zostawia `is_running=True`, a przycisk „Run evaluation” zostaje wyłączony do końca sesji.
19. **Ryzyko błędnego dopasowania modelu:** w `backend/evaluator.py:run_evaluation` rozmiar modelu jest dopasowywany przez `startswith`, więc `qwen2.5` może trafić w `qwen2.5:7b` zamiast `qwen2.5:1.5b`.
20. **Brak cache zbioru:** `app.py` wywołuje `load_unified_dataset` przy każdym rerunie, bez `st.cache_data`. Przy „all” to parsowanie ~23 tys. rekordów JSON po każdym kliknięciu.
21. **Ciche założenie w kompatybilności:** `ensure_report_compatibility` domyślnie przyjmuje `true="valid"`, gdy brakuje etykiety.
22. **Martwy kod:** `config.BINARY_LABEL_MAP`, `config.DEFAULT_MODELS`, `EvaluationRun.to_dataframe`, `utils.load_latest_results`, pakiet `frontend/views/` oraz shim `frontend/styles.py` nie są nigdzie używane.
23. **Nieaktualny profil sprzętu:** `hardware_profile.ram_available_gb` pochodzi ze skanu przy starcie sesji, nie z momentu runu.

## 5. Rekomendacje (według priorytetu)

| # | Co | Wysiłek | Warstwa | Pokrywa plan profilowania? |
|---|---|---|---|---|
| 1 | Pomiar pamięci (`/api/ps` size/size_vram, RSS drzewa procesów Ollamy, CPU, VRAM), tryb warm/cold, blok `hardware` z wersją Ollamy i OS | M | backend + docs (+ drobne UI) | tak (cały) |
| 2 | Podanie modelowi pól `Sender` i `URL` (np. nagłówki `From:` / `Links:`), obcinanie z zachowaniem linków, osobna flaga i statystyka obcięć | S–M | backend + docs | nie |
| 3 | Manifest odtwarzalności w JSON: `dataset_choice`, `sample_seed`, ID źródłowe (`No.`/`Original_ID`/wiersz CSV), digest, kwantyzacja i `parameter_size` z `/api/show`, wersja Ollamy, `git rev-parse HEAD` + flaga „dirty”, hash tekstu promptu, `seed` w opcjach | S | backend | częściowo (wersja Ollamy, sprzęt) |
| 4 | Tryb wsadowy: lista modeli × wariantów × N powtórzeń, uruchamiana bez kliknięć (CLI `python -m backend.batch` lub kolejka w Run) | M | backend + UI | nie |
| 5 | Niepewność statystyczna: bootstrap CI dla F1/precision/recall w raporcie, test McNemara między parami runów, większe próby (≥ 500–1 000 na wariant w eksperymentach finalnych) | M | backend + UI | nie |
| 6 | Analiza optimum: front Pareto (F1 vs peak RAM / P95) i ranking przy ograniczeniach (np. „najlepszy F1 przy RAM ≤ 4 GB i P95 ≤ 5 s”) | M | UI (+ backend helper) | nie (wymaga danych z #1) |
| 7 | Deduplikacja SMS (i seed) z raportem liczby usuniętych rekordów; grupowanie seed+rephrased po `Original_ID`; jawne uzasadnienie wykluczenia spamu | S | backend + docs | nie |
| 8 | Metryki binarne per kanał (email / SMS) z pola `channel` oraz wariant metryk „unknown = błąd” obok obecnego | S | backend + UI | nie |
| 9 | Zapis `prompt_eval_count`, `eval_count`, `prompt_eval_duration`, `eval_duration` per rekord; tokeny/s; przepustowość z sumy latencji zamiast czasu ścianowego z narzutem UI; osobna statystyka dla nieudanych żądań | S | backend | nie |
| 10 | Wymiar „optymalizacji zasobów”: porównanie kwantyzacji tego samego modelu (Q4_K_M vs Q8_0), `num_ctx`, `num_thread`, CPU vs GPU | M | backend (opcje) + docs | nie |
| 11 | Precision przy realistycznym odsetku zagrożeń w ruchu (np. 1%, 5%) oraz koszt FP/FN | S | backend + UI | nie |
| 12 | Oznaczenie / odfiltrowanie runów legacy w Compare (np. domyślnie tylko nowe); runy z 100% unknown nie mogą być „najlepsze” | S | UI | nie |
| 13 | Poprawki dokumentacji: wszystkie cytaty z punktu 15 | S | docs | częściowo (sekcje 7–9) |
| 14 | Testy jednostkowe w repo (parser, metryki, sampler, mapowanie kanałów, kompatybilność raportów) i lock-file / przypięte wersje; usunięcie `matplotlib`/`seaborn` | M | backend + repo | nie |
| 15 | Bugi: „select all”, `is_running` w `try/finally`, dokładne dopasowanie nazwy modelu, `st.cache_data` dla zbioru | S | UI + backend | nie |
| 16 | Baseline nie-LLM (TF-IDF + regresja logistyczna lub prosty klasyfikator słownikowy) na tych samych próbkach jako punkt odniesienia | M | backend | nie |

### Uwaga do literatury (pkt 10 zlecenia)

Projekt (zero-shot / few-shot / CoT, binarna detekcja, lokalne modele kwantyzowane) dobrze wpisuje się w typowe prace o LLM do detekcji phishingu. Recenzent może jednak oczekiwać kilku rzeczy, których brakuje:
- **punkt odniesienia:** baseline klasyczny (ML lub fine-tuned BERT) albo duży model referencyjny (np. API), żeby pokazać, ile tracą małe modele lokalne;
- **krzywe ROC/PR:** wymagają score'ów, np. logprobów etykiety, a teraz są tylko twarde etykiety;
- **odporność na parafrazy:** zbiór rephrased z `Original_ID` do tego wprost służy, ale nie jest wykorzystany analitycznie (seed vs jego warianty);
- **różnice między zbiorami:** analiza zachowania modeli na różnych zbiorach (cross-dataset);
- **ryzyko kontaminacji:** publiczne zbiory mogły trafić do danych treningowych modeli.

## 6. Pytania do mnie

1. Na którym sprzęcie powstaną wyniki do pracy? Dokumentacja i `config.py` mówią o GTX 1060 (Pascal), a obecna maszyna ma Ryzen 7 5700U bez karty NVIDIA. Od tego zależy, czy pomiary VRAM mają sens i jak definiować „ograniczone zasoby”.
2. Jak definiujesz „środowisko o ograniczonych zasobach” w pracy (np. laptop 8–16 GB RAM bez GPU, limit pamięci modelu)? Potrzebuję tego do reguły wyboru optimum.
3. Jaka jest docelowa wielkość próby i liczba powtórzeń w eksperymentach finalnych? Czy pełny zbiór seed+sms (9 020 rekordów) jest realny czasowo na CPU (szacunkowo 10–25 h na model × wariant)?
4. Czy wykluczenie spamu z obu zbiorów to świadoma decyzja, uzgodniona z promotorem? Wpływa na interpretację FPR: spam w realnym ruchu istnieje.
5. Czy promotor oczekuje baseline'u nie-LLM lub porównania z dużym modelem chmurowym?
6. Czy rephrased (generowany przez LLM) ma być w ewaluacji głównej, czy tylko jako test odporności na parafrazy?
7. Czy któreś z sześciu runów legacy mają trafić do pracy? Jeśli tak, trzeba je wyraźnie opisać jako inny protokół (stary prompt, `num_ctx` 2048, bez warm-upu).
8. Czy kwantyzacja (Q4 vs Q8 tego samego modelu) ma być osią eksperymentu „optymalizacji zasobów”, czy tylko rozmiar modelu?