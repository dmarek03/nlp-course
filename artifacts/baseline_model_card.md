# Karta modelu: baseline TF-IDF + regresja logistyczna (PolEmo 2.0-IN)

Wersja artefaktu: 1.0 (2026-10-18). Źródło: notebook `lab_2/002_Baseline.ipynb` (Laboratorium 2). Plik modelu: `artifacts/final_pipe.joblib`. Opis danych: `artifacts/dataset_card.md`.

## Pipeline

Jeden obiekt `sklearn.pipeline.Pipeline` (scikit-learn 1.9.1, Python ≥ 3.13), wejście to surowy string, bez dodatkowego preprocessingu:

```python
Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                              max_features=50_000, sublinear_tf=True)),
    ("logreg", LogisticRegression(C=5.5, max_iter=1000, random_state=42)),
])
```

- Reprezentacja: TF-IDF na słowach i bigramach, 50 000 cech, macierz rzadka (gęstość ok. 0,22%). Domyślnie `lowercase=True`, tokeny jednoznakowe są odrzucane.
- Klasyfikator: wieloklasowa regresja logistyczna, `C=5.5` wybrane na zbiorze walidacyjnym (siatka 0.1–10).
- Wczytanie: `joblib.load("artifacts/final_pipe.joblib")`, potem `predict(texts)` albo `predict_proba(texts)`.

## Co przewiduje

Sentyment **całego tekstu** po polsku, jedna z 4 klas: `plus` (pozytywny), `minus` (negatywny), `zero` (neutralny, często tekst informacyjny, a nie recenzja), `amb` (ambiwalentny: zalety i wady w jednym tekście). `predict_proba` zwraca prawdopodobieństwa w kolejności `["amb", "minus", "plus", "zero"]`.

## Dane treningowe

- PolEmo 2.0-IN z benchmarku KLEJ, wersja oczyszczona z Lab 1 (`polemo2_clean.parquet`, SHA-256 `fa27e9ec…`, zweryfikowany w notebooku). Recenzje lekarzy i hoteli oraz teksty prasowe.
- Model końcowy trenowano na **train + validation** (5782 + 723 = 6505 tekstów), już po wyborze konfiguracji.
- Tekst jest wstępnie stokenizowany (`lekarz . Bardzo`, `Była m`). Model zna tylko taki format.

## Splity i metryka

- Oficjalny podział KLEJ: train 5782 / validation 723 / test 722.
- Protokół: train do uczenia, validation do wszystkich decyzji (reprezentacja, C, wybór finalisty), test **dokładnie raz** na końcu.
- Główna metryka: **macro-F1**, bo klasy są niezbalansowane (`minus` ok. 38%, `amb` ok. 18%). Metryki pomocnicze: accuracy, macro-precision, macro-recall, F1 per klasa, 95% przedział ufności z bootstrapu (1000 prób).

## Wyniki: validation i final test

| split | trening na | accuracy | macro-F1 | uwagi |
|---|---|---:|---:|---|
| validation | train | 0.827 | 0.800 | wynik, na podstawie którego wybrano model |
| **test** | train + validation | **0.832** | **0.800** (95% CI: 0.766–0.831) | jednorazowa ocena końcowa |

F1 per klasa na teście:

| klasa | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| `amb` | 0.600 | 0.444 | 0.511 | 108 |
| `minus` | 0.815 | 0.923 | 0.866 | 300 |
| `plus` | 0.864 | 0.838 | 0.851 | 197 |
| `zero` | 1.000 | 0.949 | 0.974 | 117 |

Punkty odniesienia (validation): `Dummy(stratified)` macro-F1 0.231, `Dummy(most_frequent)` 0.136. Wynik na teście jest praktycznie równy walidacyjnemu, więc nic nie wskazuje na przeuczenie wyboru konfiguracji do zbioru walidacyjnego.

Rozważane alternatywy (validation, trening na train):

| model | macro-F1 | dlaczego nie wybrany |
|---|---:|---|
| LinearSVC + TF-IDF `char_wb` (1, 5) | 0.810 | różnica w granicach CI, ok. 11× wolniejsza inferencja, brak `predict_proba` |
| FeatureUnion word + char, LogReg `class_weight="balanced"` | 0.816 | 2× większy model (6.6 MB), kilkukrotnie dłuższy trening; sprawdzony dopiero po wyborze finalisty |

## Czas treningu i inferencji

Pomiar na laptopie (macOS, CPU, jeden proces):

| | wartość |
|---|---:|
| trening całego pipeline'u (train + validation, 6505 tekstów) | ok. 3.0 s |
| inferencja (mediana z 3 wywołań `predict` na 722 tekstach testu) | ok. 0.08 ms / dokument |
| rozmiar pliku `.joblib` | 3.46 MB |

Czasy różnią się między uruchomieniami o kilkadziesiąt procent; ważne są rzędy wielkości.

## Co wynika z zaobserwowanych porażek

Galerię błędów, analizę długości i wagi cech liczono dla wariantu z domyślnym `C=1` (ta sama reprezentacja, mocniejsza regularyzacja). Test odporności i porównanie z modelem znakowym dotyczą konfiguracji końcowej (`C=5.5`, trening na train).

- **`amb` to najtrudniejsza klasa** (F1 0.51 na teście). Najczęstszy błąd to `amb` → `minus`: model wybiera dominującą polaryzację, a słownictwo negatywne ma większe wagi. W galerii błędów (15 przykładów) 9 to `MIXED_SENTIMENT`, a 2 to `AMBIGUOUS_GOLD`, gdzie etykieta `amb` jest dyskusyjna, a predykcja modelu uzasadniona.
- **Krótkie teksty są trudniejsze:** najkrótszy kwartyl (28–71 słów) ma macro-F1 0.748, a koszyk 101–133 słów 0.842. Model na n-gramach znakowych radzi sobie z krótkimi tekstami wyraźnie lepiej (0.801 vs 0.735).
- **Wrażliwość na pisownię:** usunięcie polskich znaków obniża macro-F1 z 0.800 do 0.727, literówki w 10% słów do 0.789. Wielkość liter nie ma wpływu.
- **Skróty (shortcuts) w wagach:** klasa `zero` jest rozpoznawana głównie po słownictwie prasowym (`pap`, `proc`, `prof`, `polsce`), a nie po braku sentymentu. Unigram `nie` ma bardzo dużą wagę dla `minus`, a wzmacniacz `bardzo` dla `plus`. Wysokie wagi mają też słowa domenowe (`lekarza`, `doktora`, `śniadania`) i pozostałości pre-tokenizacji (`em`).
- **Wniosek:** model dobrze rozpoznaje jednoznaczne opinie, ale nie rozumie kontrastu („miła, ale…”), negacji w kontekście ani ironii, a część jego skuteczności wynika z gatunku i domeny tekstu.

## Czy próg confidence ma sens

Tak, jako mechanizm „nie wiem, przekaż człowiekowi”. Na walidacji (wariant z domyślnym `C=1`):

| próg max(proba) | pokrycie | accuracy na przyjętych | macro-F1 na przyjętych |
|---:|---:|---:|---:|
| brak | 100% | 0.827 | 0.793 |
| 0.4 | 92% | 0.856 | 0.809 |
| 0.5 | 72% | 0.929 | 0.844 |
| 0.6 | 56% | 0.951 | 0.807 |
| 0.7 | 38% | 0.974 | 0.740 |

- Rozsądny punkt pracy to próg ok. **0.5**: model odpowiada na ok. 3/4 tekstów z accuracy ok. 0.93, a resztę kieruje do człowieka.
- Przy progach ≥ 0.7 accuracy dalej rośnie, ale macro-F1 spada, bo wśród przyjętych przykładów prawie nie zostaje `amb`. Model rzadko jest pewny przy tekstach ambiwalentnych, a to właśnie w nich najczęściej się myli.
- Ograniczenia: tabelę policzono dla wariantu `C=1`, a nie dla modelu końcowego (`C=5.5`). Słabsza regularyzacja daje zwykle bardziej skrajne prawdopodobieństwa, więc próg trzeba dobrać ponownie na walidacji. Kalibracji prawdopodobieństw nie sprawdzano, więc próg 0.5 nie oznacza „50% szans na poprawność”.

## Do czego model może być użyty

- Szybka klasyfikacja sentymentu polskich recenzji **lekarzy i hoteli** w formacie zgodnym z PolEmo (stokenizowany tekst z polskimi znakami).
- Statystyki zbiorcze (np. udział opinii negatywnych w czasie) i wstępne sortowanie opinii do przeglądu, najlepiej z progiem confidence i weryfikacją człowieka dla niepewnych przypadków.
- Mocny, tani i interpretowalny **punkt odniesienia** dla większych modeli (transformerów, LLM-ów): nowy model powinien wyraźnie pokonać macro-F1 0.800 na tym samym teście.

## Do czego nie powinien być używany

- **Decyzje o konkretnych osobach lub obiektach**, np. ocena lekarza, ranking hoteli, moderacja kont. Sentyment opinii to nie jakość usługi, a błędy modelu skupiają się w klasie `amb`.
- **Inne domeny i gatunki** (produkty, media społecznościowe, czaty, wiadomości) i inne języki: model nauczył się słownictwa i skrótów specyficznych dla PolEmo.
- **Tekst surowy, bez pre-tokenizacji albo bez polskich znaków**: format wejścia różni się od treningowego, a brak znaków obniża wynik o ok. 0.07 macro-F1.
- **Analiza aspektów, zdań lub ironii**: etykieta dotyczy całego tekstu, a model nie modeluje kontrastu ani kontekstu.
- **Wiarygodna klasyfikacja `amb`** bez nadzoru człowieka (recall tylko 0.44).
