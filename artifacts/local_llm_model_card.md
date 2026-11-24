# Karta modelu: lokalny LLM `gemma3:4b-it-q8_0` (klasyfikacja sentymentu i ekstrakcja informacji po polsku)

Wersja artefaktu: 1.0 (2026-11-21). Źródło: notebook `lab_6/006_Local_LLM_Model_Zoo.ipynb` (Laboratorium 6). Opis środowiska: `artifacts/local_llm_environment.json`. Opis danych: `artifacts/dataset_card.md`. Baseline: `artifacts/baseline_model_card.md`. Środowisko: Ollama 0.40.14, Python 3.13.12.

Model rekomendowany do dalszych laboratoriów: **`gemma3:4b-it-q8_0` (Gemma 3, 4B, kwantyzacja Q8_0)**, uruchamiany lokalnie przez Ollamę w trybie zero-shot. Porównywane w laboratorium: `gemma3:4b` (Q4_K_M, ta sama architektura, niższa kwantyzacja), `gemma3:1b` (Q4_K_M, mniejszy model) oraz klasyczny baseline TF-IDF + regresja logistyczna.

## Model

| | `gemma3:4b-it-q8_0` (rekomendowany) | `gemma3:4b` (porównanie) | `gemma3:1b` (porównanie) |
|---|---|---|---|
| rodzina | Gemma 3 (Google DeepMind) | Gemma 3 | Gemma 3 |
| liczba parametrów (według Ollamy) | ok. 3.88 mld | ok. 3.88 mld | ok. 1.00 mld |
| kwantyzacja | **Q8_0** | Q4_K_M | Q4_K_M |
| rozmiar pliku | 4.65 GB | 3.12 GB | 0.75 GB |
| pamięć po załadowaniu (`/api/ps`) | 4.13 GB | 2.09 GB | 0.81 GB |
| źródło | biblioteka Ollamy | biblioteka Ollamy | biblioteka Ollamy |


## Środowisko

| | |
|---|---|
| system | macOS 26.7.1 (Darwin 25.6.0) |
| CPU | Apple M4 Max, 16 rdzeni |
| RAM | 64 GB |
| GPU / VRAM | zintegrowane GPU Apple, pamięć unifikowana współdzielona z RAM (brak osobnej puli VRAM) |
| Ollama | 0.40.14 |

Ustawienia generowania wspólne dla wszystkich modeli (`GEN_OPTIONS`): `temperature=0`, `seed=42`, `num_ctx=4096`. Limit `num_predict`: 16 w klasyfikacji, 128 w zadaniu strukturalnym.

## Zadanie

Dwa zadania, oba po polsku, bez dostrajania modelu:

1. **Klasyfikacja sentymentu** całego tekstu do 4 klas: `negatywna`, `pozytywna`, `neutralna`, `niejednoznaczna` (PolEmo 2.0-IN). Instrukcja wspólna dla wszystkich modeli (jako komunikat systemowy): „Oceń wydźwięk opinii. Odpowiedz wyłącznie jedną etykietą, bez cudzysłowów, pogrubień i uzasadnienia. Dozwolone etykiety: negatywna, pozytywna, neutralna, niejednoznaczna.” Odpowiedź interpretuje wyłącznie funkcja `parse_label`; wszystko inne to `INVALID` i liczy się jako błąd.
2. **Ekstrakcja informacji do JSON** (osoba, organizacja, lokalizacja) na 6 krótkich zdaniach, w trzech trybach: `free` (tylko instrukcja), `json` (`format="json"`) i `schema` (`format` ze schematem JSON).

## Dane

- **Klasyfikacja:** 400 dokumentów (100 na klasę) wylosowanych deterministycznie (`seed=42`) ze zbioru walidacyjnego PolEmo 2.0-IN, zapisanych w `artifacts/lab6_classification_sample.parquet`. Odpowiedzi modeli są zapisane w `artifacts/lab6_classification_<model>.parquet` (dla `gemma3:1b` i `gemma3:4b`).
- **Ekstrakcja:** 6 ręcznie przygotowanych przypadków (`IE_CASES`), w tym jeden bez żadnej encji.

## Szybkość i pamięć

Pomiary w stanie ciepłym (mediana z 5 powtórzeń po rozgrzewce), krótki prompt, 128 tokenów:

| | `gemma3:4b-it-q8_0` | `gemma3:4b` | `gemma3:1b` |
|---|---:|---:|---:|
| TTFT | 0.045 s | 0.037 s | ok. 0.017 s |
| tokeny/s | 57.6 | 64.2 | ok. 207 |
| mediana czasu na dokument w klasyfikacji | 224 ms | 222 ms | ok. 42–106 ms (zależnie od uruchomienia) |

Cold start (czas ładowania, 1., 2. i 3. powtórzenie) zmierzono tylko dla Q4_K_M: `gemma3:4b` 2.33 / 1.10 / 1.12 s, `gemma3:1b` 1.30 / 0.54 / 0.53 s. 1. powtórzenie jest ok. 2× wolniejsze, bo plik trzeba wczytać z dysku, a kolejne korzystają z pamięci podręcznej systemu. Dla Q8_0 cold startu nie mierzono. Czasy zmieniają się między uruchomieniami o kilkanaście procent (np. `gemma3:4b`: 208 i 222 ms na dokument w dwóch przebiegach), a pomiary TTFT i tokenów/s zależą od długości promptu.

## Wyniki klasyfikacji (400 dokumentów)

| model | macro-F1 | 95% CI (bootstrap po dokumentach) | odsetek niepoprawnych odpowiedzi |
|---|---:|---|---:|
| **`gemma3:4b-it-q8_0`** | **0.494** | 0.449–0.539 | 0.0 |
| `gemma3:4b` (Q4_K_M) | 0.456 | 0.413–0.492 | 0.0 |
| `gemma3:1b` (Q4_K_M) | 0.349 | 0.318–0.382 | 0.0025 (1 z 400) |

**Różnica kwantyzacji (Q8_0 minus Q4_K_M), analiza sparowana na tych samych dokumentach:** macro-F1 +0.038, 95% CI różnicy [+0.010, +0.070] (nie zawiera zera, więc różnica jest statystycznie istotna). Obie kwantyzacje dały tę samą predykcję dla 92.25% dokumentów (369 z 400); spośród 31 dokumentów z różną predykcją 15 zmieniło się z błędnych na poprawne, a 7 z poprawnych na błędne (netto +8 dokumentów, ok. +2 punkty procentowe accuracy). Przedziały ufności obu modeli osobno nachodzą na siebie, więc wniosek o istotności wynika dopiero z analizy sparowanej.

**F1 per klasa:** dla Q8_0 nie zapisano. Dla `gemma3:4b` (Q4_K_M), który daje te same predykcje w 92% przypadków: `negatywna` 0.730, `pozytywna` 0.694, `neutralna` 0.379, `niejednoznaczna` 0.019. Dla `gemma3:1b`: 0.595 / 0.677 / 0.123 / 0.0. Poprawnych odpowiedzi na 100 dokumentów danej klasy (Q4_K_M, `gemma3:4b`): 100 / 93 / 29 / 1.

## Wyniki wyjścia strukturalnego (6 przypadków)

Poprawność składni JSON, zgodność ze schematem, poprawność wszystkich pól jednego rekordu i średni czas odpowiedzi. W trybie `schema` składnię wymusza dekoder, więc nie jest ona miarą jakości modelu.

| model | tryb | poprawny JSON | zgodny ze schematem | poprawny cały rekord |
|---|---|---:|---:|---:|
| `gemma3:4b-it-q8_0` | `json` | 1.0 | 1.0 | 0.5 |
| `gemma3:4b` (Q4_K_M) | `free` | 0.0 (ścisłe `json.loads`) | 0.0 | 0.0 (po usunięciu znaczników markdown: 0.5) |
| `gemma3:4b` (Q4_K_M) | `json` | 1.0 | 1.0 | 0.5 |
| `gemma3:4b` (Q4_K_M) | `schema` | 1.0 | 1.0 | 0.5 |
| `gemma3:1b` (Q4_K_M) | `json` / `schema` | 1.0 | 1.0 | 0.0 |

Dla Q8_0 zmierzono tylko tryb `json` (w porównaniu kwantyzacji); tryby `free` i `schema` podano dla Q4_K_M jako najbliższy punkt odniesienia. W trybie `free` oba modele zawsze owijają odpowiedź w blok kodu markdown z oznaczeniem `json`, więc ścisła składnia wychodzi 0.0 przy poprawnej treści. Wymuszenie formatu nie poprawia treści: dla `gemma3:4b` poprawność pól jest taka sama we wszystkich trybach (0.72), a dla `gemma3:1b` po wymuszeniu jest niższa (0.28 wobec 0.56 w `free` po usunięciu znaczników).

## Porównanie z baseline'em

Baseline: TF-IDF + regresja logistyczna (konfiguracja z `artifacts/baseline_model_card.md`, `C=5.5`), **wytrenowana ponownie tylko na zbiorze `train`**. Model z `final_pipe.joblib` nie został użyty, bo trenowano go na train + validation, a próbka pochodzi ze zbioru walidacyjnego (wynik 1.0 byłby przeciekiem).

| system | macro-F1 (95% CI) | ms/dokument | rozmiar na dysku | pamięć po załadowaniu | poprawność formatu |
|---|---:|---:|---:|---:|---:|
| TF-IDF + regresja logistyczna | 0.793 [0.753, 0.830] | 0.064 | 3.46 MB | ok. 52 MB | nie dotyczy |
| LLM A: `gemma3:4b-it-q8_0` | 0.494 [0.449, 0.539] | 224 | 4.65 GB | 4.13 GB | 1.0 |
| LLM B: `gemma3:4b` (Q4_K_M) | 0.456 [0.413, 0.492] | 222 | 3.12 GB | 2.09 GB | 1.0 |

Baseline ma macro-F1 wyższe o ok. 0.30, jest ok. 3500× szybszy i zajmuje ok. 80× mniej pamięci niż Q8_0. Porównanie nie jest symetryczne: baseline uczył się na 5782 oznaczonych tekstach, a LLM działa zero-shot bez żadnego przykładu.

## Błędy (analiza 20 przypadków)

Analiza obejmuje 12 przypadków z klasyfikacji i 8 z zadania strukturalnego, wybranych z odpowiedzi `gemma3:1b` i `gemma3:4b` (Q4_K_M); odpowiedzi Q8_0 nie były w niej uwzględnione. Rozkład typów: `WRONG_LABEL` 9, `INCONSISTENT` 5, `FORMAT` 2, `HALLUCINATION` 2, `OTHER` 2. Tabela jest ilustracją wzorców, a nie oszacowaniem ich częstości.

- **Zła etykieta (klasyfikacja).** Dominujący błąd. Modele wybierają dominujący ton i prawie nigdy nie przewidują klasy `niejednoznaczna` (`gemma3:4b`: 5 predykcji na 400, 1 trafiona); teksty informacyjne (`neutralna`) często uznają za pozytywne (54 ze 100 dokumentów u `gemma3:4b`). W 119 z 400 dokumentów oba mniejsze modele wskazały tę samą błędną etykietę. Pojawiają się też pomyłki przy negacji („nie można niczego zarzucić”) i ironii.
- **Brak literalnego kopiowania (ekstrakcja).** Modele sprowadzają nazwy do mianownika mimo polecenia o tej samej formie gramatycznej („Poznań” zamiast „Poznaniu”, „Polski Związek Narciarski” zamiast „Polskiego Związku Narciarskiego”), a stanowisko bywa brane za osobę („Prezes Orlenu”). Tryb `schema` tego nie zmienia.
- **Format.** W trybie `free` znaczniki markdown wokół JSON; `gemma3:1b` dodatkowo wpisuje wartości-śmieci (napis „null”, „nieznany”) i rozbija nazwy własne.

## Zastosowanie

**Rekomendowane do:** lokalnych, offline'owych prototypów klasyfikacji lub ekstrakcji informacji po polsku, gdy nie ma oznaczonych danych treningowych albo zadanie lub schemat wyniku często się zmieniają (wystarczy zmiana promptu), a priorytetem jest jakość, a nie szybkość. Wyniki w formacie JSON trzeba wymuszać (`format` ze schematem) i walidować. Wybór Q8_0 zamiast Q4_K_M ma sens, gdy pamięci jest dużo: za niewielką, ale statystycznie istotną poprawę (+0.038 macro-F1) płaci się 2× większym zużyciem pamięci i ok. 10% wolniejszym generowaniem.

**Należy wybrać inne podejście, gdy:**
- zadanie jest stałe, są dane treningowe, a dane wejściowe są podobne do treningowych: dla klasyfikacji sentymentu TF-IDF z regresją logistyczną jest lepszy (0.793 wobec 0.494 macro-F1), ok. 3500× szybszy i wielokrotnie mniejszy;
- liczy się mała pamięć i szybka odpowiedź: `gemma3:1b` (0.75 GB, ok. 2× szybsze generowanie niż 4B), ale jej jakość jest niska (macro-F1 0.349, w tym prawie nigdy nie rozpoznaje klas `neutralna` i `niejednoznaczna`);
- pomyłki są kosztowne, np. gdy trzeba poprawnie wyłapać każdą encję: trzeba użyć modelu zadaniowego (NER) albo dodać weryfikację człowieka.

## Uzasadnienie wyboru

`gemma3:4b-it-q8_0` osiągnął najwyższe macro-F1 spośród testowanych LLM-ów (0.494, CI 0.449–0.539), a sparowana analiza z Q4_K_M potwierdza, że przewaga nie jest przypadkowa (CI różnicy [+0.010, +0.070]). Poprawność formatu odpowiedzi to 1.0, a koszt jest do przyjęcia na tym sprzęcie (64 GB pamięci, ok. 0.22 s na dokument). To wybór „najlepszy z testowanych LLM-ów”, a nie najlepszy system do tego zadania: przy dostępnych danych treningowych lepszy jest baseline. Na komputerze z mniejszą pamięcią rozsądną alternatywą jest Q4_K_M (3.12 GB, te same predykcje w 92% przypadków).

## Ograniczenia

- **Danych:** jeden zbiór (PolEmo 2.0-IN), dwie domeny (recenzje lekarzy i hoteli oraz teksty prasowe), próbka 400 dokumentów (po 100 na klasę, więc nie odzwierciedla rzeczywistego rozkładu klas), zbiór walidacyjny. Zadanie strukturalne ma tylko 6 przypadków, więc jeden przypadek to 16.7 punktu procentowego.
- **Eksperymentu:** zero-shot z jednym, zamrożonym promptem (bez dostrajania osobno dla każdego modelu); jedno uruchomienie, `temperature=0`, jeden seed; przedział ufności uwzględnia tylko losowość doboru dokumentów. Prompt nie zawiera definicji klas, więc modele nie mogą wiedzieć, co zbiór rozumie przez `niejednoznaczna`. Pomiary czasu i pamięci na jednym komputerze (Apple M4 Max, 64 GB), zmienne między uruchomieniami; TTFT i tokeny/s dla krótkiego promptu. Dla Q8_0 nie mierzono cold startu, F1 per klasa ani trybów `free` i `schema`, a analiza 20 przypadków go nie obejmuje.
- **Czego wyniki nie pozwalają stwierdzić:** że lokalne LLM-y są ogólnie gorsze od modeli klasycznych (porównanie jest nierówne: zero-shot kontra uczenie na 5782 przykładach); że wynik utrzyma się przy innym prompcie, innej dziedzinie lub na sprzęcie z mniejszą pamięcią; jak zachowuje się model przy zmianie dziedziny, bo tego w laboratorium nie mierzono.
