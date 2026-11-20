# Financial Markets Data

> **Optional resources.** Teams may use the provided synthetic data, select any public dataset, create their own fictional data, or use no external data. No source, technology, method, or interface is required.

A Financial Twin explores *conditional decisions and consequences*, not a guaranteed future. All examples are educational and do not constitute financial advice.

Optional market, housing and credit indicators. This is not a trading recommendation or live-price feed.

## 1. US 10-year Treasury yield

- **Provider:** FRED
- **Access:** DIRECT CSV
- **Download or export:** https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10
- **Optional Financial Twin use:** Yield stress testing
- **Interpretation:** US rate, not a Swiss mortgage quote

## 2. USD per CHF exchange rate

- **Provider:** FRED
- **Access:** DIRECT CSV
- **Download or export:** https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXSZUS
- **Optional Financial Twin use:** Multi-currency scenarios
- **Interpretation:** Check quote convention before conversion

## 3. USD per EUR exchange rate

- **Provider:** FRED
- **Access:** DIRECT CSV
- **Download or export:** https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU
- **Optional Financial Twin use:** Cross-border scenarios
- **Interpretation:** Check quote convention

## 4. US house price index

- **Provider:** FRED / FHFA
- **Access:** DIRECT CSV
- **Download or export:** https://fred.stlouisfed.org/graph/fredgraph.csv?id=USSTHPI
- **Optional Financial Twin use:** Historical housing sensitivity
- **Interpretation:** US, not Swiss property prices

## 5. BIS residential property prices

- **Provider:** BIS
- **Access:** EXPORT INTERFACE (bulk CSV)
- **Download or export:** https://data.bis.org/bulkdownload
- **Optional Financial Twin use:** Cross-country housing price scenarios
- **Interpretation:** Select Residential property prices

## 6. BIS central bank policy rates

- **Provider:** BIS
- **Access:** EXPORT INTERFACE (bulk CSV)
- **Download or export:** https://data.bis.org/bulkdownload
- **Optional Financial Twin use:** Borrowing cost stress
- **Interpretation:** Select Central bank policy rates

## 7. BIS credit to non-financial sector

- **Provider:** BIS
- **Access:** EXPORT INTERFACE (bulk CSV)
- **Download or export:** https://data.bis.org/bulkdownload
- **Optional Financial Twin use:** Debt environment scenarios
- **Interpretation:** Aggregate credit, not personal debt

## 8. Swiss exchange rates and yields

- **Provider:** SNB
- **Access:** EXPORT INTERFACE (CSV/JSON)
- **Download or export:** https://data.snb.ch/en
- **Optional Financial Twin use:** Swiss currency and rate scenarios
- **Interpretation:** Select table and copy generated export/API link


## Source and usage checks

Check observation period, units, geographic coverage, data revisions, missing values, and the provider’s licensing/attribution terms. Historical statistics are not individual financial facts or guaranteed future assumptions.
