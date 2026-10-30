"""Build provenance-preserving Leng cohorts and DOI-overlap evidence from raw audit files."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\WORK\DeltaRisingFoundation\BioChar")
RAW = ROOT / "Datasets"
OUT = ROOT / "cleaned_data"


def normalise_doi(value: object) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    value = str(value).strip().lower()
    if not value:
        return pd.NA
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value)
    return value.rstrip(". ") or pd.NA


def count_by_doi(frame: pd.DataFrame, doi_column: str) -> pd.Series:
    return frame.dropna(subset=[doi_column]).groupby(doi_column).size()


def main() -> None:
    OUT.mkdir(exist_ok=True)

    leng_path = RAW / "BioChemical" / "42773_2022_183_MOESM9_ESM.xlsx"
    leng = pd.read_excel(leng_path).copy()
    leng.insert(0, "raw_source_row", range(2, len(leng) + 2))
    leng.insert(1, "source_package", "BioChemical Leng")
    leng.insert(2, "source_file", str(leng_path.relative_to(ROOT)).replace("\\", "/"))
    leng["source_reference_raw"] = leng["Ref."].replace(r"^\s*$", pd.NA, regex=True)
    leng["source_doi_raw"] = leng["DOI"].replace(r"^\s*$", pd.NA, regex=True)
    leng["source_reference"] = leng["source_reference_raw"].ffill()
    leng["source_doi"] = leng["source_doi_raw"].ffill().map(normalise_doi)
    leng["provenance_fill_method"] = "forward-fill within worksheet sequence; validate before external merging"
    leng["duplicate_of_raw_source_row"] = pd.NA

    raw_columns = [c for c in leng.columns if c not in {
        "raw_source_row", "source_package", "source_file", "source_reference_raw", "source_doi_raw",
        "source_reference", "source_doi", "provenance_fill_method", "duplicate_of_raw_source_row"
    }]
    duplicate_mask = leng.duplicated(subset=raw_columns, keep="first")
    row_hash = pd.util.hash_pandas_object(leng[raw_columns], index=False)
    first_rows_by_hash = {}
    for index, key in row_hash.items():
        if key not in first_rows_by_hash:
            first_rows_by_hash[key] = int(leng.loc[index, "raw_source_row"])
        elif duplicate_mask.loc[index]:
            leng.loc[index, "duplicate_of_raw_source_row"] = first_rows_by_hash[key]
    leng["exact_duplicate_flag"] = duplicate_mask
    leng["data_quality_status"] = "retain"
    leng.loc[duplicate_mask, "data_quality_status"] = "exclude_from_model_exact_duplicate"
    leng.to_csv(OUT / "leng_master_provenance.csv", index=False)

    targets = {
        "yield": "Yield-char (%)",
        "biochar_n": "N-char (%)",
        "ssa": "SSA-char (m2/g)",
    }
    cohort_summary = []
    for cohort_name, target in targets.items():
        cohort = leng.loc[leng[target].notna()].copy()
        cohort["cohort_name"] = f"leng_{cohort_name}"
        cohort["target_field"] = target
        cohort["model_include"] = ~cohort["exact_duplicate_flag"]
        cohort["exclusion_reason"] = pd.NA
        cohort.loc[~cohort["model_include"], "exclusion_reason"] = "exact duplicate of earlier raw source row"
        cohort.to_csv(OUT / f"leng_{cohort_name}_cohort.csv", index=False)
        cohort_summary.append({
            "cohort": f"leng_{cohort_name}",
            "target": target,
            "records_with_target": len(cohort),
            "model_included_records": int(cohort["model_include"].sum()),
            "exact_duplicates_excluded": int((~cohort["model_include"]).sum()),
            "missing_source_doi_after_fill": int(cohort["source_doi"].isna().sum()),
        })
    pd.DataFrame(cohort_summary).to_csv(OUT / "leng_cohort_summary.csv", index=False)

    experimental_refs = pd.read_excel(RAW / "experimental-run" / "SI_Data_References.xlsx").copy()
    experimental_refs["source_doi"] = experimental_refs["Web Page Link"].map(normalise_doi)
    experimental_data = pd.read_excel(RAW / "experimental-run" / "SI_Data (1).xlsx")
    experimental_data.insert(0, "raw_source_row", range(2, len(experimental_data) + 2))
    experimental_data.insert(1, "source_package", "experimental-run")
    experimental_refs["experimental_reference_id"] = experimental_refs["#"]
    exp_id_to_doi = experimental_refs.set_index("experimental_reference_id")["source_doi"]
    experimental_data["source_doi_ref1"] = experimental_data["References 1"].map(exp_id_to_doi)
    experimental_data["source_doi_ref2"] = experimental_data["References 2"].map(exp_id_to_doi)
    experimental_data["source_doi"] = experimental_data["source_doi_ref1"].fillna(experimental_data["source_doi_ref2"])

    production = pd.read_csv(RAW / "Biochar_production" / "Data_biochar_production_characteristics_202404.csv")
    production.insert(0, "raw_source_row", range(2, len(production) + 2))
    production.insert(1, "source_package", "Biochar production")
    production["source_doi"] = production["DOI"].map(normalise_doi)

    leng_counts = count_by_doi(leng, "source_doi")
    exp_ref_counts = count_by_doi(experimental_refs, "source_doi")
    exp_row_counts = count_by_doi(experimental_data, "source_doi")
    production_counts = count_by_doi(production, "source_doi")
    all_dois = sorted(set(leng_counts.index) | set(exp_ref_counts.index) | set(production_counts.index))
    ledger = pd.DataFrame({"source_doi": all_dois})
    ledger["leng_rows"] = ledger["source_doi"].map(leng_counts).fillna(0).astype(int)
    ledger["experimental_reference_rows"] = ledger["source_doi"].map(exp_ref_counts).fillna(0).astype(int)
    ledger["experimental_data_rows_ref1"] = ledger["source_doi"].map(exp_row_counts).fillna(0).astype(int)
    ledger["production_rows"] = ledger["source_doi"].map(production_counts).fillna(0).astype(int)
    ledger["overlap_flag"] = (ledger[["leng_rows", "experimental_reference_rows", "production_rows"]] > 0).sum(axis=1) > 1
    ledger["resolution_status"] = "not_overlapping"
    ledger.loc[ledger["overlap_flag"], "resolution_status"] = "block_extension_merge_pending_row_level_match"
    ledger["required_match_keys"] = "DOI; feedstock; temperature; residence time; heating rate; target value; source row"
    ledger.to_csv(OUT / "doi_overlap_ledger.csv", index=False)

    overlap = ledger[ledger["overlap_flag"]].copy()
    overlap.to_csv(OUT / "doi_overlap_ledger_overlaps_only.csv", index=False)

    leng_dois = set(leng_counts.index)
    for frame, filename in [
        (experimental_data, "experimental_run_extension_eligibility.csv"),
        (production, "biochar_production_extension_eligibility.csv"),
    ]:
        doi_columns = [c for c in frame.columns if c.startswith("source_doi")]
        frame["doi_overlap_with_leng"] = frame[doi_columns].isin(leng_dois).any(axis=1)
        frame["extension_include"] = frame["source_doi"].notna() & ~frame["doi_overlap_with_leng"]
        frame["extension_exclusion_reason"] = pd.NA
        frame.loc[frame["source_doi"].isna(), "extension_exclusion_reason"] = "missing DOI or reference mapping"
        frame.loc[frame["doi_overlap_with_leng"], "extension_exclusion_reason"] = "source DOI represented in Leng; excluded conservatively"
        frame.to_csv(OUT / filename, index=False)

    pd.DataFrame([
        {
            "extension_source": "experimental-run",
            "total_records": len(experimental_data),
            "eligible_after_doi_screen": int(experimental_data["extension_include"].sum()),
            "excluded_doi_overlap": int(experimental_data["doi_overlap_with_leng"].sum()),
            "excluded_missing_provenance": int(experimental_data["source_doi"].isna().sum()),
        },
        {
            "extension_source": "Biochar production",
            "total_records": len(production),
            "eligible_after_doi_screen": int(production["extension_include"].sum()),
            "excluded_doi_overlap": int(production["doi_overlap_with_leng"].sum()),
            "excluded_missing_provenance": int(production["source_doi"].isna().sum()),
        },
    ]).to_csv(OUT / "extension_doi_screen_summary.csv", index=False)


if __name__ == "__main__":
    main()
