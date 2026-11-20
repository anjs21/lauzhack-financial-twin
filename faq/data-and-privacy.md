# FAQ — Data and Privacy

**LauzHack 2026 | AI Financial Twin**

### 1. Are the provided financial profiles real?
No. They are fictional, synthetic records.

### 2. Can we use real personal banking transactions?
Do not use real confidential or identifiable personal financial records for this challenge.

### 3. Can we use public economic data?
Yes, where access and intended use comply with provider terms. The `resources/data/` folder links to optional downloadable datasets and export interfaces.

### 4. Is public data mandatory?
No. Teams may use only synthetic data, generate their own, or add appropriately licensed public data.

### 5. Is data on a public website automatically free to redistribute?
No. Access, downloading, reuse, and redistribution can have different conditions. Check each dataset's license.

### 6. Can we download inflation or exchange-rate data?
Yes, from suitable public providers. Record the indicator, unit, date range, retrieval date, and source.

### 7. Are Swiss national averages suitable for Alexia?
They may inform assumptions, but they do not establish Alexia's individual income or expenses. Treat them as context, not personal facts.

### 8. Can we generate additional profiles?
Yes. Preserve fictional identifiers, coherent financial relationships, and clear generation assumptions.

### 9. Can we add personal preferences or goals?
Yes, as synthetic or explicitly user-entered demo parameters. Do not infer sensitive characteristics unnecessarily.

### 10. How should we connect the CSV files?
Use the documented `profile_id` and other keys. Check the actual CSV headers and the data dictionary before coding.

### 11. What if some fields are missing?
Ask for inputs, use labelled synthetic assumptions, or mark outputs unavailable. Do not silently invent facts.

### 12. Can we train a model on the synthetic dataset?
Yes, subject to relevant licenses. Explain that synthetic patterns may not generalize to real households.

### 13. Are the provided figures representative of Switzerland?
Not necessarily. Synthetic plausibility is not the same as statistical representativeness.

### 14. Can we include third-party datasets in our GitHub repository?
Only if redistribution is permitted. Otherwise link to the provider and document how to retrieve the data.

### 15. What should we document?
Source, license, geography, dates, units, transformations, synthetic generation, and known limitations.

### 16. Can we commit API keys?
No. Use local environment variables or secret-management facilities and keep credentials out of Git history.
