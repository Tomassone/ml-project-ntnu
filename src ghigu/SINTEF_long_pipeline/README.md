# Pipeline Tokke–Vinje: wide → long

## Cartelle consigliate

```
sintef_project/
├── data/
│   ├── raw/             # 7 CSV SINTEF + Tokke_Vinje_topology.yaml
│   └── processed/       # output generato, NON da committare
├── notebooks/
│   └── 01_build_feature_table.ipynb
└── src/
    └── build_long_dataset.py
```

Il notebook è **autosufficiente**: non necessita del file `src/build_long_dataset.py`.
Lo script `.py` è un'alternativa per eseguire la stessa pipeline senza Jupyter.

## Uso in VS Code
1. Colloca i file SINTEF nella cartella `data/raw/` (nomi originali).
2. Installa `pip install -r requirements.txt` nel virtualenv del progetto.
3. Apri `01_build_feature_table.ipynb` e seleziona il kernel del virtualenv.
4. Esegui le celle per la prova sui primi 2 casi e per il test di reversibilità.
5. Imposta `RUN_FULL_BUILD=True` nella cella finale; genera Parquet in `data/processed/`.

## Script alternativo

```
python src/build_long_dataset.py --data-dir data/raw --out data/processed/feature_table_long.parquet
```

Prova rapida in CSV: `--limit-cases 2 --format csv --out data/processed/test.csv`.

## Definizioni
- `run_no`, `generator`, `hour` identificano univocamente la singola predizione.
- `target` è l'etichetta ON/OFF, un valore per riga.
- `timestamp = starttime + hour`, UTC.
- Volumi del caso = valori iniziali a `starttime`.
- Water value = valore terminale a `starttime + 168h` come indicato da SINTEF.
- I reservoir upstream includono i direct e vengono deduplicati tramite la topologia.
- Water value medio pesato = media dei water value terminali pesata per i volumi iniziali.
- Le colonne `min_volume_*` e `min_flow_*` sono conservate come input globali con il valore orario originale; la scelta finale e la mappatura dei vincoli ai generatori appartengono all'EDA.
- `total_calculation_time` non è mai una feature: se fornito, è esclusivamente metadata per verificare il round trip.

## Precauzioni per il training futuro
- Le finestre di 168 ore partono **ogni giorno** e quindi si sovrappongono. Evita split casuali per riga.
- Valuta se gli inflow all'interno della settimana siano disponibili/previsti al momento della decisione; altrimenti sono leakage.
- Valuta le feature aggiuntive (andamento dei prezzi, lag, vincoli attivi) solo nella fase EDA/feature engineering.
- `gen_eff_curve` e `turb_eff_curves` sono curve e non possono essere ridotte a un numero di efficienza senza un'ipotesi operativa; non sono state inserite in questa prima struttura.
- Per il test finale senza `total_calculation_time`, `to_wide` riproduce i nomi delle colonne target originali lasciando questa colonna a `NaN` se non c'è metadata.
