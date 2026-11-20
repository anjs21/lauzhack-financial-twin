# Macroeconomic Data

> **Optional resources.** Teams may use the provided synthetic data, select any public dataset, create their own fictional data, or use no external data. No source, technology, method, or interface is required.

A Financial Twin explores *conditional decisions and consequences*, not a guaranteed future. All examples are educational and do not constitute financial advice.

Optional economic environmental conditions for simulation, historical replay, and sensitivity analysis.

## 1. Swiss inflation

- **Provider:** World Bank
- **Access:** CSV ZIP
- **Download or export:** https://api.worldbank.org/v2/country/CHE/indicator/FP.CPI.TOTL.ZG?downloadformat=csv
- **Optional Financial Twin use:** Cost-of-living scenarios
- **Interpretation:** Annual rate

## 2. Swiss GDP growth

- **Provider:** World Bank
- **Access:** CSV ZIP
- **Download or export:** https://api.worldbank.org/v2/country/CHE/indicator/NY.GDP.MKTP.KD.ZG?downloadformat=csv
- **Optional Financial Twin use:** Macroeconomic context
- **Interpretation:** Not a personal income model

## 3. Swiss unemployment

- **Provider:** World Bank
- **Access:** CSV ZIP
- **Download or export:** https://api.worldbank.org/v2/country/CHE/indicator/SL.UEM.TOTL.ZS?downloadformat=csv
- **Optional Financial Twin use:** Job-loss stress conditions
- **Interpretation:** National aggregate

## 4. US consumer price index

- **Provider:** FRED / US BLS
- **Access:** DIRECT CSV
- **Download or export:** https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL
- **Optional Financial Twin use:** US inflation experiments
- **Interpretation:** US index, not Swiss

## 5. US unemployment rate

- **Provider:** FRED / US BLS
- **Access:** DIRECT CSV
- **Download or export:** https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE
- **Optional Financial Twin use:** US labour-market experiments
- **Interpretation:** US rate, not Swiss

## 6. US federal funds rate

- **Provider:** FRED
- **Access:** DIRECT CSV
- **Download or export:** https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS
- **Optional Financial Twin use:** Interest-rate sensitivity
- **Interpretation:** US policy environment

## 7. Euro area and European GDP

- **Provider:** Eurostat
- **Access:** EXPORT INTERFACE (dataset nama_10_gdp)
- **Download or export:** https://ec.europa.eu/eurostat/databrowser/view/nama_10_gdp/default/table?lang=en
- **Optional Financial Twin use:** Cross-country economic comparison
- **Interpretation:** Select series, units and geography

## 8. European unemployment

- **Provider:** Eurostat
- **Access:** EXPORT INTERFACE (dataset une_rt_m)
- **Download or export:** https://ec.europa.eu/eurostat/databrowser/view/une_rt_m/default/table?lang=en
- **Optional Financial Twin use:** Relocation scenarios
- **Interpretation:** Select series and download

## 9. BIS international statistics

- **Provider:** Bank for International Settlements
- **Access:** EXPORT INTERFACE (bulk CSV)
- **Download or export:** https://data.bis.org/bulkdownload
- **Optional Financial Twin use:** Cross-country interest and price stress
- **Interpretation:** Choose named dataset from catalogue


## Source and usage checks

Check observation period, units, geographic coverage, data revisions, missing values, and the provider’s licensing/attribution terms. Historical statistics are not individual financial facts or guaranteed future assumptions.
