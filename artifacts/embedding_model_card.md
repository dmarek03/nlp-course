# Karta modelu embeddingowego: multilingual-e5-small (PolEmo 2.0-IN)

Wersja artefaktu: 1.0 (2026-10-22). Źródło: notebook `lab_3/003_From_Tokens_to_Embeddings.ipynb` (Laboratorium 3). Opis danych: `artifacts/dataset_card.md`. Model rekomendowany do dalszych laboratoriów: **E5**. Porównywany z TF-IDF i `paraphrase-multilingual-MiniLM-L12-v2`.

## Model

- **Checkpoint:** `intfloat/multilingual-e5-small` (Hugging Face), ładowany przez `SentenceTransformer` (sentence-transformers 6.1.0, transformers 5.20.2, torch 2.14.1).
- **Wymiar reprezentacji:** 384, typ `float32`, wektor gęsty.
- **Normalizacja:** L2 (`encode(..., normalize_embeddings=True)`), więc cosinus = iloczyn skalarny.
- **Okno modelu:** `max_seq_length = 512` tokenów; dłuższe teksty są obcinane bez ostrzeżenia.
- **Wymagane prefiksy wejścia:**
  - zadania symetryczne (podobieństwo dwóch tekstów, sąsiedzi, PCA): `query: ` po obu stronach;
  - wyszukiwanie: `query: ` dla zapytania, `passage: ` dla dokumentów.
  Bez prefiksów model działa gorzej niż opisano w tej karcie; prefiks zużywa też kilka tokenów okna.

## Dane

- **Zbiór:** PolEmo 2.0-IN (KLEJ), wersja oczyszczona z Lab 1 (`polemo2_clean.parquet`); recenzje lekarzy i hoteli oraz teksty prasowe (klasa `zero`). Tekst jest wstępnie stokenizowany (`lekarz . Bardzo`, `Była m`).
- **Wielkość próbki:** 1000 dokumentów, **tylko ze splitu train** (validation i test nie były używane).
- **Dobór próbki:** `train_test_split(train_df, train_size=1000, stratify=target, random_state=42)`, czyli deterministyczna próbka zachowująca proporcje klas: `minus` 380, `plus` 271, `amb` 181, `zero` 168.
- **Odsetek tekstów obciętych przez okno:** **4.1%** (mediana 198 tokenów po tokenizacji z prefiksem). Dla porównania MiniLM (okno 128) obcina 76.9% tekstów.

## Koszt

Pomiar na laptopie (macOS, GPU Apple `mps`), bez ładowania modelu, po jednej rozgrzewce na 20 tekstach:

| reprezentacja | czas [s] (1000 dok.) | ms/dokument | wymiary | rozmiar [MB] | typ |
|---|---:|---:|---:|---:|---|
| TF-IDF (1–2-gramy) | 0.20 | 0.20 | 15 633 | 1.13 | rzadka |
| MiniLM | 1.57 | 1.57 | 384 | 1.46 | gęsta |
| **E5** | **3.75** | **3.75** | **384** | **1.46** | gęsta |

E5 jest najdroższy (ok. 19× wolniej niż TF-IDF, ok. 2.4× wolniej niż MiniLM), głównie przez 4× dłuższe okno. Rozmiar w pamięci to stałe 384 × 4 B = ok. 1.5 KB na dokument. Czasy różnią się między uruchomieniami o kilkanaście procent.

## Jakość sąsiedztwa

Średnia purity@k po całej próbce (sąsiedzi bez samego dokumentu, cosinus, wersja `query:`):

| reprezentacja | purity@1 | purity@5 | purity@10 |
|---|---:|---:|---:|
| **E5** | **0.694** | **0.667** | **0.642** |
| MiniLM | 0.608 | 0.587 | 0.561 |
| TF-IDF | 0.582 | 0.555 | 0.534 |
| losowi sąsiedzi | 0.279 | 0.279 | 0.279 |

purity@5 per klasa (E5 / losowy punkt odniesienia = udział klasy): `zero` 0.829 / 0.168, `minus` 0.751 / 0.380, `plus` 0.661 / 0.271, `amb` **0.348** / 0.181. E5 jest najlepszy w każdej klasie, ale dla `amb` tylko o ok. 0.17 powyżej losowego.

## Wyszukiwanie

Miniaturowy benchmark: 8 zapytań (po 2 na klasę), relewancja = etykieta sentymentu; dokumenty z `passage:`, zapytania z `query:`.

| reprezentacja | Precision@5 | Precision@10 |
|---|---:|---:|
| **E5** | **0.575** | **0.512** |
| MiniLM | 0.600 | 0.550 |
| TF-IDF | 0.600 | 0.525 |

E5 per klasa zapytań (P@5 / P@10): `minus` 0.8 / 0.7, `plus` 0.6 / 0.75, `zero` 0.5 / 0.35, `amb` 0.4 / 0.25. Różnice między reprezentacjami są mniejsze niż 0.1, więc przy 8 zapytaniach **nie wskazują zwycięzcy**.

## Analiza jakościowa

- **PCA (2 składowe, 17.6% wariancji: PC1 12.3%, PC2 5.3%):** klasa `zero` tworzy osobny obszar (wysokie PC2), zgodnie z tym, że są to głównie teksty prasowe. `plus`, `minus` i `amb` tworzą dwa duże skupiska po obu stronach PC1 i w obu są całkowicie wymieszane. Skupiska odpowiadają najpewniej domenie (lekarze / hotele), a nie sentymentowi; środki ciężkości tych trzech klas leżą w pustej przestrzeni między skupiskami. Wykres 2D pokazuje mały wycinek przestrzeni 384-wymiarowej.
- **Dominujące typy błędów (12 przypadków z Zadania 8: 6 z sąsiadów i wyszukiwania, 6 par kontrastowych):**
  - **wspólny temat + różny sentyment:** np. dwie recenzje o leczeniu migreny, rozczarowana i wdzięczna, mają cosinus 0.945; zapytanie negatywne o pokój zwraca na 1. miejscu pozytywną recenzję pokoju (0.917);
  - **negacja:** „Hotel jest dobry.” / „Hotel nie jest dobry.” = 0.918, czyli **powyżej** pary niezwiązanej tematycznie (0.834, Δ +0.084);
  - **podobna forma / wspólne słownictwo:** zamiana ocen aspektów („Pokój czysty, ale obsługa fatalna” / „Pokój brudny, ale obsługa wspaniała”) daje najwyższe podobieństwo w tabeli (0.964).
  - Δ E5 względem pary niezwiązanej mieści się w zakresie 0.04–0.13 dla wszystkich 12 przypadków: skala cosinusów E5 jest skupiona wysoko, więc wartości bezwzględnych nie należy porównywać z innymi modelami.

## Zastosowanie

**Rekomendowany do:**
- wyszukiwania semantycznego dokumentów (krótkie zapytanie → długi dokument) i jako retriever w przyszłym systemie **RAG**, z prefiksami `query:` / `passage:`;
- grupowania i wyszukiwania tekstów **według tematu** (domena, sytuacja, obiekt recenzji);
- reprezentacji długich recenzji: okno 512 tokenów obejmuje w całości ok. 96% tekstów PolEmo.

**Nie wybierać wyłącznie na podstawie tego laboratorium do:**
- **klasyfikacji sentymentu:** wysoka purity nie zastępuje ewaluacji klasyfikatora; trzeba wytrenować model na embeddingach i porównać macro-F1 z baseline'em TF-IDF + LogReg z Lab 2 (0.800 na teście);
- **rozróżniania negacji i przeciwnych ocen tego samego obiektu** oraz klasy `amb`;
- **progów podobieństwa** typu „cosinus > 0.9 = to samo znaczenie”: niezwiązane zdania mają ok. 0.83;
- **wyszukiwania po dokładnych nazwach i terminach:** TF-IDF jest tu równie dobry (P@5 0.600) i ok. 19× tańszy.

## Ograniczenia

- **Danych:** 1000 dokumentów z jednego zbioru i dwóch domen (lekarze, hotele) plus teksty prasowe; tylko język polski, tekst wstępnie stokenizowany. Klasa `zero` różni się gatunkiem tekstu, co zawyża jej purity. Wyniki nie przenoszą się na inne domeny ani na surowy tekst.
- **Benchmarku:** wyszukiwanie oceniono na 8 ręcznie napisanych zapytaniach, a relewancję zastępuje etykieta sentymentu, więc P@k zależy głównie od doboru zapytań (dla E5 P@10 waha się od 0.25 dla zapytań `amb` do 0.75 dla `plus`). Purity mierzy tylko zgodność etykiet sentymentu, a nie ogólną jakość semantyczną. Porównanie z MiniLM jest częściowo porównaniem ilości przeczytanego tekstu (obcinanie 4.1% vs 76.9%). Analiza jakościowa obejmuje 12 przypadków.
- **Samej reprezentacji:** embedding koduje przede wszystkim temat i formę wypowiedzi, a słabo ocenę: negacja i zamiana ocen zmieniają cosinus minimalnie. Skala podobieństw jest skompresowana (0.83–0.96 dla wszystkich badanych par). Tekst dłuższy niż 512 tokenów jest obcinany, więc werdykt na końcu bardzo długiej recenzji może zostać pominięty. Model wymaga poprawnych prefiksów; pomyłka (np. `passage:` w porównaniu symetrycznym) zmienia wyniki.
