# DoTA dataset

W paperze przedstawiony jest zbiór danych ***Detection of Traffic Anomaly*** (DoTA). Zbiór zawiera 4677 nagrań z kamerki umieszczonej za przednią szybą samochodu, w trakcie których przedstawione są różne scenariusze anomalii występujących na drogach. Zbiór zawiera etykiety określające:
* które klatki w nagraniu są anomalią,
* jakie obiekty były zaangażowane w fakt dojścia do anomalii,
* jaki typ anomalii wystąpił.

Nagrania oryginalnie występowały w 30fps natomiast autorzy pobrali klatki dla częstotliwości 10fps. Rozdzielczość filmów to 1280x720. 

### Zasady etykietowania:
Labelowanie takich zbiorów jest bardzo subiektywne (odnosząc się chociaż do tego kiedy dana anomalia się zaczęła a kie dy skończyła), dlatego w celu zapewnienia najlepszej jakości etykiet każdy film był labelowany przez trzy osoby. Klatki są labelowane jako anomalia w momencie gdy dojście do anomalii jest nieuniknione. Koniec anomalii występuje w momencie gdy wszystkie obiekty biorące udział w anomalii znikną z pola widzenia lub zatrzymają się.

# Klasy anomalii:
Twórcy datasetu rozróżniają 9 kategorii anomalii:
| ID  | Skrót | Kategoria anomalii                                                               |
|-----|-------|----------------------------------------------------------------------------------------|
| 1   | ST    | Kolizja z pojazdem, który rusza, zatrzymuje się lub stoi                 |
| 2   | AH    | Kolizja z pojazdem jadącym z przodu lub oczekującym                                |
| 3   | LA    | Kolizja z pojazdem poruszającym się w tym samym kierunku                 |
| 4   | OC    | Kolizja z innym nadjeżdżającym pojazdem                                                |
| 5   | TC    | Kolizja z pojazdem, który wjeżdża na drogę lub przez nią przejeżdża                     |
| 6   | VP    | Kolizja pomiędzy pojazdem a pieszym                                              |
| 7   | VO    | Kolizja z przeszkodą na drodze                                             |
| 8   | OO    | Utracenie panowania nad pojazdem i zjechanie z jezdni na jej lewą lub prawą część                           |
| 9   | UK    | Różne                                                                               |

W paperze przetestowano jak radzą sobie autoencodery. Podzielono typy uczenia nienadzorowanego na dwa typy: ***frame-level*** oraz ***object-oriented***. 

### Frame-level:
* W ramach tego pierwszego przetestowano klasyczny **ConvAE** z funkcją straty MSE. Stworzono wersje działające na zdjęciu w Grayscale oraz na gęstych optycznych przepływach (jako input model przyjmuje mapt 30x227x227 z pretrenowanego FlowNet2). 
* Stworzono również **ConvLSTMAE** - enkoder CNN najpierw pobiera informacje przestrzenne z każdej klatki a następnie wielowarstwowy ConvLSTM rekurencyjnie enkoduje cechy czasowe. Następnnie dekoder CNN rekonstruuje wejściowe klatki. Dla tego wariantu również utworzono wersje działające na Grayscale i optical flow. 
* Stworzono również **AnoPred**, który bierze cztery poprzednie klatki RGB jako input, przepuszcza je przez UNet w celu przewidzenia przyszłej klatki. AnoPred wzmacnia dokładność predykcji przez wielozadaniową stratę włączającą intensywność zdjęcia, optyczne przepływy, gradienty oraz straty adwersarzowe. Rozwiązanie to zostało zaproponowane dla monitoringu miejskiego. Z racji tego zaproponowano również kombinację AnoPred z Mask-RCNN aby przewidywać tylko obiekty a nie całe sceny.

### Object-cetric:
* **TAD** modeluje trajektorie bounding boxów w ruchu drogowym przy pomocy wielostrumieniowego RNN enkodera-dekodera do zdekodowania przeszłych trajektorii i własnego ruchu i do przewidywania przyszłych bounding boxów obiektów. Wyniki predykcji są zbierane a następnie zamiast mierzyć accuracy mierzymy anomaly score.
* Zaproponowano również traktować normalność obiektu jako multimodalną i użyć *k-means* by znaleźć normalne klastry w przestrzeni ukrytej. Zaproponowano również użycie Margin Learning'u (ML) aby wyegzekwować duże odległości pomiędzy normalnymi i nienormalnymi cechami. W ten sposób powstał **TAD+ML** - wykorzystuje *k-means* do klasteryzacji stanów ukrytych enkodera. Każdy klaster jest uważany za jedną normalność, np.: jeden typ normalnego ruchu więc każda próbka treningowa jest inicjalizowana ID klastra jako rodzaj normalności. Potem używamy center loss by wyegzekwować ciasne rozkłady próbek z tej samej "normalności" i by zmusić próbki z innych rozkładów do bycia rozróżnialną. Center loss jest znacznie bardziej wydajny niż triplet loss w wielkich zbiorach treningowych.
**Ensamble** - połączenie AnoPred+Mask z TAD+ML w metodę zespołową. Każda metoda została wytrenowana oddzielnie i skupiono się na uśrednianiu anomaly score. Zauważono, że taka póżna fuzja jest lepsza niż łączenie ukrytych cech we wczesnej fazie i trenowanii modeli razem, ponieważ ich cechy ukryte są skalowane w różny sposób. AnoPred+Mask enkoduje jedną cechę na klatkę w czasie gdy TAD+ML ma jedą cechę na obiekt.

