# Karta modelu: transformer do klasyfikacji sentymentu (PolEmo 2.0-IN)

Wersja artefaktu: 1.0 (2026-10-31). Źródło: notebook `lab_4/004_Transformer_as_a_Model.ipynb` (Laboratorium 4). Opis danych: `artifacts/dataset_card.md`. Baseline: `artifacts/baseline_model_card.md`. Środowisko: transformers 5.21.0, torch 2.14.1, scikit-learn 1.9.1, Python 3.13.

Model finalny (wybrany na walidacji): **fine-tuned HerBERT**. Model bazowy laboratorium: wielojęzyczny DistilBERT.

## Model

| | model bazowy | model polski (Zadanie 8, finalista) |
|---|---|---|
| checkpoint | `distilbert/distilbert-base-multilingual-cased` | `allegro/herbert-base-cased` |
| architektura | DistilBERT (enkoder, model destylowany), `DistilBertForSequenceClassification` | BERT (enkoder), `BertForSequenceClassification` |
| warstwy / głowy uwagi / wymiar ukryty | 6 / 12 / 768 | 12 / 12 / 768 |
| liczba parametrów (z głową na 4 klasy) | 135 327 748 (sam enkoder 134 734 080, ok. 514 MB w FP32) | 124 445 956 (ok. 475 MB w FP32) |
| głowa klasyfikacyjna | `pre_classifier` (768→768) + `classifier` (768→4) na wektorze `[CLS]`, nowo zainicjalizowane | `classifier` (768→4) na wyjściu poolera, nowo zainicjalizowana |

W DistilBERT-cie ok. 68% parametrów to tablica embeddingów (119 547 × 768). Model wczytywany z `attn_implementation="eager"`, bo implementacja SDPA na MPS nie obsługuje dropoutu w treningu.

## Tokenizator

- **Checkpoint:** ten sam co model (`distilbert-base-multilingual-cased`: WordPiece, słownik 119 547; `herbert-base-cased`: BPE, słownik 50 000).
- **`MAX_LENGTH = 128`.** Decyzja: zostawiamy 128 jako kompromis kosztowy (czas treningu i pamięć na MPS), świadomie akceptując obcinanie końca tekstu.
- **Procent tekstów obciętych przy 128** (cały `train`, tokenizer DistilBERT-a): **84.3%** (mediana 222.5 tokenu, p90 416, p95 524, maksimum 1807).
- **Porównanie z HerBERT-em** (500 losowych tekstów z `train`):

| tokenizer | mediana | p90 | % obciętych przy 128 | tokeny na słowo |
|---|---:|---:|---:|---:|
| DistilBERT multilingual | 215 | 398 | 83.6% | 2.17 |
| HerBERT | 145 | 272 | 59.2% | 1.46 |

- **Padding i obcinanie:** `truncation=True, max_length=128` przy tokenizacji (koniec tekstu jest ucinany); padding dynamiczny przez `DataCollatorWithPadding` do najdłuższego tekstu w batchu (bez `padding="max_length"`).

## Zadanie

Klasyfikacja sentymentu całego tekstu do 4 klas: `amb` (ambiwalentny), `minus` (negatywny), `plus` (pozytywny), `zero` (neutralny / tekst informacyjny).

- `label2id = {"amb": 0, "minus": 1, "plus": 2, "zero": 3}`
- `id2label = {0: "amb", 1: "minus", 2: "plus", 3: "zero"}`

## Dane

- PolEmo 2.0-IN (KLEJ), wersja oczyszczona z Lab 1 (`polemo2_clean.parquet`), oficjalny podział.
- Liczebność: **train 5782, validation 723, test 722**.
- Bez ograniczania liczby przykładów z powodów sprzętowych (pełny `train`).
- Model finalny trenowany na **train + validation (6505 tekstów)**; test użyty jednokrotnie.

## Trening

| | frozen | fine-tuned DistilBERT | fine-tuned HerBERT | HerBERT finalny |
|---|---|---|---|---|
| dane treningowe | train | train | train | train + validation |
| learning rate | 1e-3 | 5e-5 | 5e-5 | 5e-5 |
| parametry trenowalne | 593 668 (0.44%) | 135 327 748 (100%) | 124 445 956 (100%) | 124 445 956 (100%) |

Wspólne: batch size 16, 3 epoki, `weight_decay=0.01`, warmup 10% kroków (`warmup_steps=108`, dla train + validation 122; w transformers 5.21 nie ma argumentu `warmup_ratio`), seed 42, ewaluacja po każdej epoce na validation (dla modelu finalnego wyłączona), `save_strategy="no"`.

## Wyniki

Validation (modele trenowane na `train`):

| model | accuracy | macro-F1 | 95% CI (bootstrap) |
|---|---:|---:|---:|
| TF-IDF + Linear SVM | 0.822 | 0.788 | 0.758–0.817 |
| frozen DistilBERT | 0.678 | 0.581 | 0.557–0.606 |
| fine-tuned DistilBERT | 0.786 | 0.768 | 0.738–0.797 |
| fine-tuned HerBERT | 0.862 | 0.844 | 0.817–0.872 |

Różnice macro-F1 (sparowany bootstrap na tych samych przykładach): HerBERT − DistilBERT **+0.076** (CI +0.045…+0.107), HerBERT − SVM **+0.056** (CI +0.023…+0.090), DistilBERT − SVM −0.020 (CI −0.056…+0.013, nieistotna).

**Finalny wynik testowy** (trening na train + validation, jedna ocena na teście):

| model | accuracy | macro-F1 | 95% CI | F1 `amb` | F1 `minus` | F1 `plus` | F1 `zero` |
|---|---:|---:|---:|---:|---:|---:|---:|
| **HerBERT (finalny)** | **0.882** | **0.859** | 0.830–0.883 | 0.631 | 0.908 | 0.909 | 0.987 |
| TF-IDF + Linear SVM | 0.830 | 0.791 | 0.759–0.822 | 0.464 | 0.861 | 0.860 | 0.978 |

Δ macro-F1 (HerBERT − SVM) na teście: **+0.068** (CI +0.030…+0.104). Dla porównania finalista Lab 2 (TF-IDF + LogReg) na tym samym teście: macro-F1 0.800, accuracy 0.832.

## Koszt

Urządzenie: laptop macOS, GPU Apple (`mps`); TF-IDF + SVM na CPU.

| model | czas treningu | ms/dokument (walidacja, batch 16) |
|---|---:|---:|
| TF-IDF + Linear SVM | ok. 1 s | 0.06 |
| frozen DistilBERT | 31 s | 1.47 |
| fine-tuned DistilBERT | 303 s | 1.38 |
| fine-tuned HerBERT | 351 s | 2.84 |
| HerBERT finalny (train + validation) | 389 s | 3.06 (test) |

Czasy różnią się między uruchomieniami o kilka–kilkanaście procent.

## Błędy

Analiza 12 losowych błędów HerBERT-a (trenowanego tylko na `train`) na walidacji:

- **TRUNCATION (4):** decydująca ocena (często podsumowanie „z czystym sumieniem polecam”) leży poza oknem 128 tokenów; 5 z 12 tekstów było obciętych.
- **MIXED_SENTIMENT (3):** zalety i wady w jednym tekście; model wybiera dominującą polaryzację.
- **LEXICAL_TRAP (3):** słownictwo o przeciwnym wydźwięku niż ocena (negatywne słowa w obronie lekarza, pozytywne o innym lekarzu, ocena liczbowa „4,5/5”).
- **AMBIGUOUS_GOLD (2):** etykieta dyskusyjna, predykcja uzasadniona.
- Wszystkie 12 błędów dotyczy klasy `amb` (jako etykiety lub predykcji); brak pomyłek `plus` ↔ `minus`. TF-IDF + Linear SVM popełnił ten sam błąd w 4 z 12 przypadków.

## Zastosowanie

**Transformer (HerBERT) jest uzasadniony**, gdy liczy się jakość klasyfikacji polskich recenzji, zwłaszcza opinii mieszanych (`amb`: F1 0.63 vs 0.46 dla SVM), a dostępny jest GPU i akceptowalna jest inferencja rzędu kilku ms na dokument.

**Prostszy baseline (TF-IDF + Linear SVM) może być lepszym wyborem**, gdy liczą się koszt, szybkość (ok. 50× szybsza inferencja, trening w sekundę na CPU), prostota wdrożenia i interpretowalność, albo gdy dostępny jest tylko model niedopasowany do języka: wielojęzyczny DistilBERT nie był lepszy od SVM przy ok. 300× dłuższym treningu.

## Ograniczenia

- **Danych:** jeden zbiór (PolEmo 2.0-IN), dwie domeny (recenzje lekarzy i hoteli) i teksty prasowe w klasie `zero`; tekst wstępnie stokenizowany (`Był em`, `miała m`), więc wyniki nie przenoszą się wprost na surowy tekst ani inne domeny; etykieta `amb` bywa subiektywna.
- **Eksperymentu:** `MAX_LENGTH=128` obcina 84% (DistilBERT) i 59% (HerBERT) tekstów; 3 epoki bez strojenia learning rate i bez wczesnego zatrzymania (strata walidacyjna DistilBERT-a rosła po 2. epoce); jeden seed, a dostrajanie ma wyraźną wariancję między uruchomieniami; analiza błędów na 12 przypadkach; czasy mierzone na jednym laptopie z MPS.
- **Czego wyniki nie pozwalają stwierdzić:** że transformery są ogólnie lepsze od modeli klasycznych (wielojęzyczny DistilBERT nie był); że przewaga HerBERT-a wynika z tokenizatora, bo modele różnią się też korpusem pretreningowym, głębokością i sposobem treningu; jak wypadłyby modele przy dłuższym oknie, więcej epokach lub innych domenach.
