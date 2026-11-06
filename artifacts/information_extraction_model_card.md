# Karta systemu: rozpoznawanie nazw własnych (PER, ORG, LOC) w tekstach polskich

Wersja artefaktu: 1.0 (2026-11-06). Źródło: notebook `lab_5/005_Information_Extraction_simplified.ipynb` (Laboratorium 5). Środowisko: transformers 5.21.0, torch 2.14.1, Python 3.13.12.

Rekomendowany system: **HerBERT NER (`pczarnik/herbert-base-ner`), bez dostrajania, bez progu pewności**. Rekomendowane zastosowanie: wykrywanie danych osobowych (osób) do anonimizacji tekstów polskich, z kontrolą człowieka. Porównywane w laboratorium: system regułowy ze słownikiem oraz LLM (Bielik) z odpowiedzią w JSON.

## System

- **Wybrane podejście:** pretrenowany model NER (klasyfikacja tokenów).
- **Model:** `pczarnik/herbert-base-ner`, źródło: Hugging Face. Jest to HerBERT (`allegro/herbert-base-cased`) dostrojony przez autora modelu na polskiej części WikiANN (etykiety IOB2: `B-PER`, `I-PER` itd.). Użyta rewizja: `8770e1d2b8c273e4dababfaa7d30a7443005ff70`, plik wag ok. 473 MB. Licencja: nie została sprawdzona w ramach laboratorium (w lokalnej pamięci podręcznej nie ma pliku README), więc przed zastosowaniem komercyjnym trzeba ją zweryfikować na stronie modelu.
- **Uruchomienie:** `pipeline("ner", model="pczarnik/herbert-base-ner", aggregation_strategy="simple")` (podsłowa łączone w całe encje). Bez progu pewności. Etykiety spoza `PER`/`ORG`/`LOC` są odrzucane, prefiks `B-`/`I-` usuwany.
- **Format zwracanych encji** (ten sam co w zbiorze referencyjnym, z dodatkowym polem `score`):

```json
{"text": "Donald Tusk", "label": "PER", "start": 0, "end": 11, "score": 0.989}
```

Konwencja zakresów: `text[start:end]`, `start` włącznie, `end` wyłącznie. Pole `text` jest wycinane z oryginalnego dokumentu (nie z pola `word` modelu), a format przechodzi audyt z Zadania 1 (zgodność `text[start:end]`, brak nakładania, poprawne etykiety).

## Zadanie

Rozpoznawanie nazw własnych w pojedynczych zdaniach polskich, z podziałem na 3 klasy:

- `PER`: osoba (imię i nazwisko, w formie występującej w tekście, np. „Jana Kowalskiego”);
- `ORG`: organizacja, firma, instytucja (np. „Komisja Europejska”, „PZPN”, „Mimo Wszystko”);
- `LOC`: lokalizacja (miasto, kraj; np. „Warszawie”, „Paryża”).

**Wytyczne anotacji** (rozstrzygnięcia przypadków granicznych występujących w zbiorze):

- Encje są płaskie, bez zagnieżdżeń: nazwa miasta wewnątrz nazwy instytucji nie jest dodatkowo oznaczana jako `LOC` (np. „Politechniki Gdańskiej” to tylko `ORG`).
- Dopełnienie lokalizacyjne wchodzi do nazwy organizacji, gdy jest częścią jej nazwy własnej („Muzeum Narodowe w Krakowie”, „Uniwersytetem Ekonomicznym w Katowicach”). W konstrukcji „AGH w Krakowie” miasto nie jest częścią nazwy, więc `AGH` to `ORG`, a „Krakowie” osobne `LOC`.
- Stanowisko, tytuł i rzeczowniki pospolite przed nazwą nie należą do encji: z „Prezydent Warszawy Rafał Trzaskowski” oznaczamy „Rafał Trzaskowski” (`PER`) i „Warszawy” (`LOC`), z „firmą IBM” samo „IBM”, z „Zespół OpenAI” samo „OpenAI”.
- Encję oznaczamy w formie, w jakiej występuje w tekście (z odmianą), a nie w mianowniku.

## Dane

Zbiór własny: 30 polskich zdań z ręcznie oznaczonymi encjami, podział według `doc_id % 3` (DEV: `doc_id % 3 == 0`).

| | dokumenty | encje | PER | ORG | LOC | encje / dokument |
|---|---:|---:|---:|---:|---:|---:|
| DEV | 10 | 25 | 7 | 12 | 6 | 2.50 |
| TEST | 20 | 52 | 13 | 22 | 17 | 2.60 |
| razem | 30 | 77 | 20 | 34 | 23 | 2.57 |

HerBERT NER nie był dostrajany na tych danych, więc DEV i TEST służą tu tylko do oceny (DEV dodatkowo do wyboru reguł i promptu pozostałych systemów). Audyt zbioru: 0 problemów.

## Ewaluacja (na TEST)

Wynik rekomendowanego systemu (20 dokumentów, 52 encje):

| | precision | recall | F1 |
|---|---:|---:|---:|
| exact | 0.980 | 0.923 | **0.950** (95% CI 0.90–0.99, bootstrap po dokumentach) |
| relaxed | 1.000 | 0.942 | 0.970 |

- **F1 per klasa (exact):** PER 1.000, ORG 0.878, LOC 1.000.
- **Błędy granic:** 1.
- **Wpływ progu pewności** (HerBERT, cały zbiór DEV + TEST, exact): bez progu F1 0.961; próg 0.70 daje najlepsze F1 0.973 (precision 1.000, recall 0.948, coverage 0.973); próg 0.90 obniża recall do 0.883 i F1 do 0.938 (coverage 0.907). Do anonimizacji nie stosujemy progu, bo każdy próg może tylko obniżyć recall.

Porównanie z pozostałymi systemami (TEST):

| system | exact F1 | 95% CI | exact recall | relaxed F1 | błędy granic | PER / ORG / LOC F1 | ms/dok | poprawność formatu |
|---|---:|---|---:|---:|---:|---|---:|---:|
| reguły + słownik | 0.659 | 0.54–0.78 | 0.558 | 0.659 | 0 | 0.79 / 0.43 / 0.74 | 0.09 | 1.00 |
| **HerBERT NER** | **0.950** | 0.90–0.99 | **0.923** | 0.970 | 1 | 1.00 / 0.88 / 1.00 | 17.0 | 1.00 |
| LLM (Bielik 4.5B, Q8_0, `temperature=0`) | 0.878 | 0.80–0.94 | 0.827 | 0.918 | 2 | 0.96 / 0.90 / 0.79 | 2418 | 1.00 (JSON), 6 encji spoza tekstu |

Przedziały ufności HerBERTa i LLM nachodzą na siebie, więc na tak małym TEST nie można uznać różnicy między nimi za rozstrzygniętą. Reguły są od obu wyraźnie gorsze.

## Koszt

- **Czas inferencji:** ok. 9–17 ms na dokument (średnia po 20 dokumentach TEST, z pominięciem pierwszego wywołania; dwa uruchomienia dały 8.7 i 17.0 ms). Pipeline uruchamiany z ustawieniami domyślnymi, bez wskazania urządzenia, na laptopie macOS (nie sprawdzano, czy obliczenia szły na CPU, czy na GPU Apple). Dla porównania: reguły ok. 0.09 ms, LLM ok. 2400 ms na dokument.
- **Wymagania:** biblioteki `transformers` i `torch`, pobranie ok. 0,5 GB przy pierwszym uruchomieniu, pierwsze wywołanie obejmuje inicjalizację modelu. Brak dodatkowego serwera; GPU nie jest wymagane. (LLM wymaga lokalnego serwera Ollama i ok. 5 GB modelu.)
- **Koszt dostosowania:** zmiana zestawu klas wymaga ponownego dostrojenia modelu na nowych oznaczonych danych.

## Błędy

HerBERT popełnił na TEST tylko 4 błędy (3 pominięcia, 1 błąd granic), wszystkie omówione w analizie 12 przypadków:

- **FN_MISSED (3):** pominięte jednowyrazowe nazwy firm: Google, Samsung (początek zdania) i OpenAI (po rzeczowniku „Zespół”). Nie pominął żadnej osoby ani lokalizacji.
- **BOUNDARY (1):** nazwa złożona ze zwykłych słów („Mimo Wszystko”) ucięta do „Mimo”.

Dla porównania, reguły: głównie `FN_MISSED` (16) i `LABEL` (7, organizacje uznane za osoby); LLM: głównie `FN_MISSED` (6), wszystkie spowodowane zwróceniem tekstu zmienionego względem zdania (inna forma odmiany, literówka), oraz 2 błędy granic i 1 błąd klasy.

## Zastosowanie

**Rekomendowane do:** wykrywania osób w tekstach polskich w celu anonimizacji, gdy pominięcie encji jest kosztowne. W naszym teście HerBERT miał najwyższy recall (0.923) i znalazł wszystkie 13 osób (PER F1 = 1.0), a jego błędy dotyczyły głównie nazw firm. Wynik należy traktować jako wstępny: opiera się na 13 encjach PER, więc przed użyciem produkcyjnym trzeba go potwierdzić na większym zbiorze z domeny docelowej, a wynik uzupełnić kontrolą człowieka (a dla danych o stałym formacie, np. PESEL czy e-mail, regułami).

**Należy wybrać inne podejście, gdy:**

- liczy się przepustowość i znane są wszystkie nazwy (np. oznaczanie znanych organizacji i lokalizacji w dużym strumieniu tekstu): reguły ze słownikiem są ok. 190 razy szybsze;
- schemat encji często się zmienia i nie ma danych do dostrajania: LLM z odpowiedzią w JSON, kosztem ok. 140 razy dłuższego czasu niż HerBERT;
- teksty pochodzą z innej domeny niż Wikipedia (prasa, pisma urzędowe): trzeba najpierw zmierzyć jakość na własnych danych.

## Ograniczenia

- **Zbioru:** 30 krótkich, jednozdaniowych tekstów o znanych osobach i instytucjach, wzorowanych na języku encyklopedycznym. TEST ma tylko 20 dokumentów i 52 encje (13 `PER`), więc przedziały ufności są szerokie (HerBERT 0.90–0.99). Zbiór nie zawiera tekstów prasowych ani urzędowych, nazw nietypowych, encji zagnieżdżonych ani przypadków niejednoznacznych. Podział DEV/TEST jest losowy (`doc_id % 3`), a część nazw (np. Warszawa, Kraków) występuje w obu częściach.
- **Metryk:** exact F1 liczone zbiorczo (micro) na jednym podziale; relaxed uznaje trafienie przy każdym nakładaniu zakresów tej samej klasy, więc ukrywa błędy granic, a przy anonimizacji ucięta nazwa zostawia fragment danych. Czasy mierzone na jednym laptopie i różnią się między uruchomieniami (8.7 vs 17.0 ms dla HerBERTa). Wynik LLM pochodzi z jednego uruchomienia, z promptem i dwoma przykładami wybranymi na DEV.
- **Systemu:** model uczył się na WikiANN, czyli automatycznie oznaczonej Wikipedii, więc zna wiele z nazw użytych w teście, a jego wyniki mogą być zawyżone w stosunku do innych domen. Adnotacje WikiANN zawierają szum i mogą się różnić od naszych wytycznych (granice nazw złożonych). Model pomijał jednowyrazowe nazwy firm. Nie obsługuje nowych klas bez ponownego dostrojenia i nie zwraca wyniku dla encji zagnieżdżonych. Wynik `score` nie jest skalibrowaną miarą pewności.
- **Czego wyniki nie pozwalają stwierdzić:** że HerBERT jest istotnie lepszy od LLM (przedziały ufności się nakładają); że wynik PER = 1.0 utrzyma się na innych danych; jak system działa na tekstach spoza domeny encyklopedycznej.
