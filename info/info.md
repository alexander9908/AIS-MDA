# blackhole
/dtu/blackhole/10/178320/preprocessed_1/final
# preprocessing 

## kilde: https://arxiv.org/pdf/2109.03958

## Filtrering 

- sog > 30 ud (sejler urealistisk hurtigt)
- Remowes ancored and mored vessels (secured stoped vessels)
- Remove if AIS signal cuts for more then 2 hours split if between 0-2 hours combine.
- Remove AIS voyages < 20 messages (4 hours)
- Remove abnormal messages if speed is above 40 knots (calculated from dist and time)
- long voyages gets split (voyage > 20 hours)

## max long, lat
- Long <
- Lat < 

ture er stapet med 5 min 


## Features in df
Columns: LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI



TrAISformer: https://github.com/CIA-Oceanix/TrAISformer



### Spørgsmål til Møde 2
- Hvilken loss function skal bruges 
- Gennemgang af preprocessing (map_reduce)

What model: 
- Transformer 
- Self supervices transformer

Kan vi få port data? 
- log, lat, name etc of all ports 

then we know where to cut trips. fx mmsi 123 goes from cph -> london -> barcalona -> nyc 

then when the messages cross a defines port border the trip gets split. 

so 3 trips from the full  cph -> london -> barcalona -> nyc 

trip 1: cph -> london
trip 2: london -> barcalona
trip 3: barcalona -> nyc 





