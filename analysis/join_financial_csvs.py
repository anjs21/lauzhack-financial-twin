"""Join the eight supplied Data CSVs. Repeated parent values must not be summed.
Run from any directory: python analysis/join_financial_csvs.py
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'Data'


def main():
    inputs = {}
    def read(filename, prefix, keys):
        frame = pd.read_csv(DATA / filename, dtype=str, keep_default_na=False)
        inputs[filename] = len(frame)
        # Preserve original values and make provenance of overlapping fields explicit.
        return frame.rename(columns={c: f'{prefix}__{c}' for c in frame.columns if c not in keys})

    profiles = read('synthetic-profiles.csv', 'profile', {'profile_id'})
    balances = read('synthetic-cash-balances.csv', 'cash', {'profile_id', 'account_id'})
    accounts = read('synthetic-accounts.csv', 'account', {'profile_id', 'account_id'})
    assets = read('synthetic-assets.csv', 'asset', {'profile_id'})
    goals = read('synthetic-goals.csv', 'goal', {'profile_id'})
    liabilities = read('synthetic-liabilities.csv', 'liability', {'profile_id'})
    assumptions = read('scenario-assumptions.csv', 'assumption', {'profile_id', 'scenario_id'})
    catalog = read('scenario-catalog.csv', 'scenario', set()).rename(columns={'scenario__id': 'scenario_id'})
    assert profiles.profile_id.is_unique
    assert catalog.scenario_id.is_unique
    assert not accounts.duplicated(['profile_id', 'account_id']).any()
    assert not balances.duplicated(['profile_id', 'account_id', 'cash__month']).any()
    profile_ids = set(profiles.profile_id)
    for frame in (balances, accounts, assets, goals, liabilities, assumptions):
        assert set(frame.profile_id) <= profile_ids, 'Orphan profile ID'
    assert set(assumptions.scenario_id) <= set(catalog.scenario_id), 'Orphan scenario ID'

    joined = profiles.merge(balances, on='profile_id', how='outer', validate='one_to_many')
    joined = joined.merge(accounts, on=['profile_id', 'account_id'], how='outer', validate='many_to_one')
    # Investment assets have separate account IDs absent from the cash-account table.
    # Join asset ownership by profile; retain asset__account_id independently.
    for frame in (assets, goals, liabilities, assumptions):
        joined = joined.merge(frame, on='profile_id', how='outer', validate='many_to_many')
    # Outer join retains all 120 catalog scenarios, including those without assumptions.
    joined = joined.merge(catalog, on='scenario_id', how='outer', validate='many_to_one')
    joined = joined.sort_values(['profile_id', 'cash__month', 'asset__asset_id', 'goal__goal_id',
                                 'assumption__assumption_id'], na_position='last', kind='stable')
    assert set(joined.profile_id.dropna()) == profile_ids
    assert set(joined.scenario_id.dropna()) == set(catalog.scenario_id)
    for source, col in [(assets, 'asset__asset_id'), (goals, 'goal__goal_id'),
                        (liabilities, 'liability__liability_id'), (assumptions, 'assumption__assumption_id')]:
        assert set(source[col]) == set(joined[col].dropna()), f'Lost records: {col}'
    assert len(joined.dropna(subset=['cash__month']).drop_duplicates(['profile_id','account_id','cash__month'])) == len(balances)
    output = DATA / 'joined-data.csv'
    joined.to_csv(output, index=False)
    metadata = {
        'inputs': inputs,
        'output': 'Data/joined-data.csv',
        'rows': len(joined), 'columns': len(joined.columns),
        'catalog_only_rows': int(joined.profile_id.isna().sum()),
        'join_keys': ['profile_id', 'profile_id + account_id for cash accounts', 'scenario_id = scenario catalog id'],
        'column_naming': 'Source prefix plus double underscore, except shared identifier keys.',
        'grain': 'Profile × month × asset × goal × liability × assumption; plus unlinked catalog rows.',
        'warnings': [
            'One-to-many joins repeat cash, income, balances, goals, and asset values. Do not sum these columns without deduplicating at the original source grain.',
            'Common currency, as_of_date, notes, and amount columns are not join keys. Their source versions are preserved separately.',
            'asset__account_id identifies the owned asset account; account_id identifies the cash account. Investment holdings are joined through profile ownership.',
            'Scenario parameters are linked inputs, not evidence the scenario was applied to the supplied cash history.',
            'Blank cells indicate missing/unlinked source records, not zero amounts.',
            'This flattened CSV is for inspection. Split profiles and reconstruct source grains before statistical analysis or model training.'
        ]
    }
    (DATA / 'joined-data-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({k: metadata[k] for k in ['output','rows','columns','catalog_only_rows']}))


if __name__ == '__main__':
    main()
