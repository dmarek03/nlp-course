# Karta zbioru: PolEmo 2.0-IN (KLEJ), wersja oczyszczona

Wersja artefaktu: 1.0 (2026-10-12). Źródło decyzji: notebook `001_Text.ipynb` (Laboratorium 1). Manifest: `artifacts/splits.json`.

## Dataset

- **Nazwa:** PolEmo 2.0-IN, zadanie *in-domain* z benchmarku KLEJ (Allegro).
- **Źródło danych:** Hugging Face `allegro/klej-polemo2-in`, pliki `train.csv`, `valid.csv`, `test.csv` pobrane z gałęzi `main` (stan na 2026-10-11; ostatnia zmiana repozytorium 2022-08-30, rewizja `9843af3`). Oryginał: CLARIN-PL, https://clarin-pl.eu/dspace/handle/11321/710. Publikacja: Kocoń, Miłkowski, Zaśko-Zielińska, *Multi-Level Sentiment Analysis of PolEmo 2.0*, CoNLL 2019, https://aclanthology.org/K19-1092/.
- **Pliki w tym repozytorium:**
  - `artifacts/polemo2_clean.parquet` - dane po czyszczeniu (7227 rekordów, 18 kolumn), SHA-256: `fa27e9ec45ded6b1ca620ca8699af58bd39f249180a1e449cf5f1c16c5fff34b`;
  - `artifacts/splits.json` - manifest: `random_state`, `label_map`, rozmiary splitów przed i po czyszczeniu, lista operacji czyszczenia, hash parqueta, wersje Pythona (3.13.12), pandas (3.0.6) i scikit-learn (1.9.1).

## Task

Klasyfikacja sentymentu **całego tekstu** do jednej z 4 klas. Wejście: kolumna `text` (recenzja), wyjście: kolumna `target`. Teksty to przede wszystkim opinie konsumenckie o lekarzach i hotelach; wariant *in-domain* oznacza, że train, validation i test pochodzą z tych samych dziedzin. Klasa `zero` zawiera w dużej części fragmenty tekstów informacyjnych i prasowych, a nie recenzje.

Metryki: benchmark KLEJ raportuje accuracy. Ze względu na niezbalansowane klasy raportujemy **też macro-F1** i wyniki per klasa. Punkty odniesienia z laboratorium:

| model | accuracy (test) | macro-F1 (test) |
|---|---|---|
| zawsze `minus` (najczęstsza klasa) | 0.416 | ok. 0.15 |
| TF-IDF (1–2-gramy) + regresja logistyczna, bez strojenia | 0.832 | 0.787 |

## Schema

Parquet ma 18 kolumn. Wiersze są w kolejności train → validation → test, tak jak w plikach źródłowych.

| kolumna | typ | grupa | opis |
|---|---|---|---|
| `text` | str | źródło | oryginalny tekst (w pliku źródłowym kolumna `sentence`) |
| `target` | str | źródło | etykieta: `plus` / `minus` / `zero` / `amb` (po `LABEL_MAP`) |
| `split` | str | źródło | oficjalny podział KLEJ: `train` / `validation` / `test` |
| `n_chars` | int | długość | liczba znaków |
| `n_words` | int | długość | liczba słów wg regexu `\w+` (liczona przez `str.findall`, bo `str.count` w pandas 3 nie rozpoznaje polskich liter) |
| `n_words_split` | int | długość | liczba tokenów po `split()`; większa od `n_words`, bo interpunkcja jest oddzielona spacjami |
| `n_lines` | int | długość | liczba linii; w tym zbiorze zawsze 1 |
| `has_url` | bool | szum | adres WWW (także zapisany ze spacjami, np. `www . strona . pl`) |
| `has_email` | bool | szum | adres e-mail (także `nazwa @ domena . pl`) |
| `has_punct_burst` | bool | szum | seria co najmniej 3 znaków `!`/`?`, także rozdzielonych spacjami (`! ! !`) |
| `has_newline` | bool | szum | znak nowej linii; w tym zbiorze zawsze `False` |
| `is_very_short` | bool | szum | mniej niż 20 znaków |
| `uppercase_ratio` | float | szum | udział wielkich liter wśród liter |
| `high_uppercase` | bool | szum | `uppercase_ratio > 0.5` |
| `text_raw` | str | duplikaty | kopia `text` (identyczna), zachowana ze względu na nazewnictwo z audytu duplikatów |
| `text_norm` | str | duplikaty | tekst po normalizacji NFKC → lowercase → zwinięcie białych znaków → trim; tylko do porównań, nie do modelowania |
| `dup_in_split` | bool | duplikaty | dokładny duplikat `text_norm` w obrębie tego samego splitu; w tej wersji wszędzie `False` |
| `random_split` | str | eksperyment | losowy split stratyfikowany z Zadania 7 (`random_state=42`); **nie używać** do treningu ani ewaluacji, jest tylko materiałem porównawczym |


## Labels

`LABEL_MAP` (etykieta źródłowa → etykieta w danych):

| źródło | etykieta | znaczenie |
|---|---|---|
| `__label__meta_plus_m` | `plus` | pozytywny |
| `__label__meta_minus_m` | `minus` | negatywny |
| `__label__meta_zero` | `zero` | neutralny |
| `__label__meta_amb` | `amb` | niejednoznaczny / mieszany (np. „lekarz kompetentny, ale rejestracja fatalna”) |

Liczebności po czyszczeniu:

| split | minus | plus | amb | zero | razem |
|---|---|---|---|---|---|
| train | 2194 | 1568 | 1049 | 971 | 5782 |
| validation | 271 | 209 | 116 | 127 | 723 |
| test | 300 | 197 | 108 | 117 | 722 |

Udział klas [%] (dane surowe; po czyszczeniu różnice poniżej 0.1 pp):

| split | minus | plus | amb | zero |
|---|---|---|---|---|
| train | 37.9 | 27.1 | 18.2 | 16.8 |
| validation | 37.5 | 28.9 | 16.0 | 17.6 |
| test | 41.6 | 27.3 | 15.0 | 16.2 |

## Splits

- Zachowany **oficjalny podział KLEJ**: 5783 / 723 / 722 (train / validation / test). Po czyszczeniu: 5782 / 723 / 722. Validation i test nie były modyfikowane, więc wyniki pozostają porównywalne z literaturą dla KLEJ.
- Test ma nieco więcej `minus` (41.6% wobec 37.9% w train) i mniej `amb` (15.0% wobec 18.2%). To cecha benchmarku, nie błąd.
- Oficjalny podział **nie jest** losowym podziałem połączonych danych: losowy split stratyfikowany o tych samych rozmiarach (`random_state=42`) przypisuje 34.4% rekordów do innego splitu niż oficjalny. Wyniki liczone na własnym, losowym podziale nie są porównywalne z wynikami na oficjalnym teście.

## Data quality

Co znaleziono (Zadania 1–6):

- **Tekst jest wstępnie stokenizowany**, nie surowy: interpunkcja oddzielona spacjami (`lekarz . Bardzo`), końcówki fleksyjne odcięte od słów (`Wykupiła m`, `wrócili śmy`), brak znaków nowej linii. Mediana to 696 znaków, 102 słowa (`\w+`) i 121 tokenów (`split()`); p95 to 1606 znaków, maksimum 5391. Tokenizatory i regexy trzeba do tego dostosować (np. `! ! !` zamiast `!!!`).
- **Brak wartości brakujących** w `text` i `target`, każdy split ma ten sam zbiór 4 etykiet.
- **1 rekord bez treści:** tekst `"P"` z etykietą `amb` (train). Usunięty.
- **Teksty ucięte w połowie słowa po znaku `&`**, np. `Marka Baccarat Hotels & Res o`, `Hotel , jak za tą kwotę ( 230 - 2 osoby & aut o` (17 tekstów; prawdopodobnie błąd obsługi encji HTML `&amp;` przy budowie zbioru). Zostawione bez zmian; to najkrótsze rekordy w zbiorze, a ich etykiety dotyczą pełnej, niewidocznej recenzji.
- **Notka moderatora w treści:** 4 teksty zawierają szablon portalu „Ocena została zmieniona na "Polecam" . Powód : Zbyt lakoniczna argumentacja…” (3 z nich to `plus`, jeden w teście). To informacja o ocenie wpisana w tekst wejściowy.
- **Flagi szumu** (676 rekordów, 9.4%, ma co najmniej jedną):

| flaga | liczba rekordów | związek z etykietą |
|---|---|---|
| `has_punct_burst` | 657 | 69% to `minus` (w całym zbiorze 38%), w `zero` prawie nie występuje; potencjalny shortcut |
| `has_url` | 11 | 8 z 11 to `zero` (teksty informacyjne); artefakt źródła |
| `high_uppercase` | 10 | za mało rekordów na wniosek |
| `has_email` | 5 | za mało rekordów na wniosek |
| `is_very_short` | 1 | rekord `"P"` (usunięty) |
| `has_newline` | 0 | teksty spłaszczone |

- **Długość zależy od klasy:** najdłuższe teksty to negatywne recenzje hoteli (5 najdłuższych to `minus`), a klasa `zero` ma najwyższą medianę długości. Długość i wykrzykniki to cechy, których model może się nauczyć zamiast treści.
- **Brak dokładnych duplikatów** przed normalizacją i po niej (0 z 7228). Prawie duplikaty: patrz *Leakage audit*.

## Cleaning

Polityka: minimalna, zgodna z Zadaniem 9. Każda operacja jest zapisana w `splits.json` z liczbą usuniętych rekordów.

Zmienione:

| operacja | usunięto |
|---|---|
| rekordy bez tekstu lub etykiety (wszystkie splity) | 0 |
| dokładne duplikaty `text_norm` w obrębie train | 0 |
| rekordy train, których `text_norm` występuje w validation lub test | 0 |
| rekordy train bez treści (`n_words < 2`), czyli `"P"` | 1 |

Dodano kolumnę `dup_in_split` (flaga, nie usuwanie); w tej wersji nie ma żadnego `True`.

Świadomie **nie** zmieniono:
- treści tekstów (bez detokenizacji, bez poprawiania ucięć po `&`);
- validation i test (oficjalne rozmiary zachowane);
- tekstów z wykrzyknikami, URL-ami, e-mailami i notką moderatora (tylko flagi lub opis w tej karcie);
- **prawie duplikatów** między splitami. Są udokumentowane niżej, ale w tej wersji pozostają w danych.

## Leakage audit

- **Dokładny overlap** (`audit_split_leakage`, klucz `text_norm`): 0 tekstów występuje w więcej niż jednym splicie, zarówno w podziale oficjalnym, jak i losowym.
- **Prawie duplikaty:** TF-IDF na n-gramach znakowych 3–5 (`char_wb`) + najbliższy *inny* sąsiad (cosinus, `k=5`), 6601 unikalnych par. Przy sim ≥ 0.9 są 3 pary, z czego 1 między splitami; przy sim ≥ 0.7 jest 7 par, z czego 4 między splitami. Po ręcznym przejrzeniu wszystkich 31 par z sim ≥ 0.5 (Zadanie 11D) prawdziwych duplikatów jest 7, a między splitami leżą 4:

| sim | rekordy (indeks `all_df`) | splity | co to jest |
|---|---|---|---|
| 1.000 | 1344 ↔ 6642 | train ↔ test | ten sam tekst o bakteriach, różni się jednym `!` |
| 0.741 | 3918 ↔ 6285 | train ↔ validation | ta sama recenzja hotelu przepisana przez autora |
| 0.718 | 752 ↔ 6597 | train ↔ test | nakładające się fragmenty tego samego komunikatu prasowego |
| 0.632 | 4877 ↔ 6214 | train ↔ validation | nakładające się fragmenty tej samej notki o hotelach |

  Para 8 ↔ 6581 (sim 0.79, train ↔ test) to **fałszywy alarm**: dwie różne recenzje, które łączy tylko notka moderatora.

## License / provenance

- **Pochodzenie:** opinie z polskich portali (recenzje lekarzy i hoteli) oraz teksty informacyjne, zebrane i oznaczone przez zespół PolEmo (Politechnika Wrocławska, CLARIN-PL); anotacja ekspercka w schemacie 2+1. Do KLEJ trafiły w formie już stokenizowanej, bez metadanych o autorze, obiekcie i dacie.
- **Licencja jest podana na trzy sposoby:**
  - metadane zbioru na Hugging Face (nagłówek YAML karty): `cc-by-sa-4.0`;
  - treść karty na Hugging Face, sekcja *License*: `CC BY-NC-SA 4.0`;
  - oryginalne źródło w CLARIN-PL: `CC BY 4.0`.
- **Decyzja:** dla kopii z Hugging Face / KLEJ, której używamy, przyjmujemy **CC BY-NC-SA 4.0**, czyli najbardziej restrykcyjną z deklaracji dystrybutora. Przy sprzecznych informacjach bezpieczniej założyć węższe uprawnienia.
- **Błędy w karcie na Hugging Face:** w tabeli rozkładu klas `minus` opisano jako *positive*, a `plus` jako *negative*, czyli odwrotnie niż w opisie zadania w tej samej karcie i niż wynika z danych (`minus` = negatywny). Karta podaje kolumnę wejściową `text`, a w plikach jest `sentence`. Jako metrykę podaje tylko accuracy.

## Limitations

- **Domena i język:** tylko polski, głównie recenzje lekarzy i hoteli. Nie wolno wnioskować o jakości modelu dla innych domen (produkty, uczelnie, media społecznościowe) ani innych języków.
- **Tokenizacja:** tekst jest stokenizowany inaczej niż tekst z produkcji. Model wytrenowany na tych danych może zachowywać się inaczej na surowym tekście, jeśli nie zastosuje się tej samej tokenizacji.
- **Poziom etykiety:** etykieta dotyczy całego tekstu. Zbiór nie nadaje się do oceny sentymentu zdań, aspektów ani fragmentów.
- **Shortcuty:** klasa `zero` częściowo pochodzi z innego źródła (teksty prasowe), negatywne recenzje są dłuższe i mają więcej wykrzykników, a 4 teksty zawierają notkę zdradzającą ocenę. Wysoki wynik może częściowo wynikać z tych cech, a nie z rozumienia treści.
- **Resztkowy wyciek:** 4 pary prawie duplikatów między splitami pozostają w danych (patrz *Leakage audit*).
- **Interpretacja:** sentyment opinii to nie jakość lekarza ani hotelu. Recenzje piszą osoby, które same zdecydowały się je napisać, a klasa `amb` zależy od subiektywnej decyzji anotatora.
