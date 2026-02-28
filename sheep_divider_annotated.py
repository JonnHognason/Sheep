#!/usr/bin/env python3  
"""
Sheep Divider – communal livestock allocation by weight.
This annotated script reads an Excel workbook containing sheets:
  - Sheep (SheepID, Sex, Weight_kg)
  - Owners (PartID, Owner, unit columns - mark and shinn)
  - Settings (key/value pairs including NumberOfParts and SeparateBySex)
It then computes and writes back:
  - Part_Assignments (which part each sheep belongs to)
  - Part_Summary (totals by sex and overall for each part)
  - Owner_Shares (BaseUnits and SharePct per owner within each part)
  - Owner_Desired (target weight per owner within each part)
  - Whole_Sheep_Allocation (whole animals assigned to owners, with deltas)
"""

import sys  
import pandas as pd  
import numpy as np  
from pathlib import Path  

SKINN_PER_MARK = 240
SKINN_PER_GYLLIN = 15

def lpt_partition(weights_list, n_bins):  
    if len(weights_list) == 0:  
        return []
    indexed = list(enumerate(weights_list))  
    indexed.sort(key=lambda x: x[1], reverse=True)  
    bin_sums = [0.0] * n_bins  
    assignment = [None] * len(weights_list)  
    for idx, w in indexed:  
        b = min(range(n_bins), key=lambda k: bin_sums[k])  
        bin_sums[b] += w  
        assignment[idx] = b  
    return assignment 


def allocate_whole_sheep_to_owners(part_id, sheep_df_part, desired_by_owner):  
    if sheep_df_part.empty:  
        rows = []  
        for row in desired_by_owner:  
            rows.append({  
                "PartID": part_id,  
                "Owner": row["Owner"],  
                "SheepIDs": "",  
                "AllocatedWeight": 0.0,  
                "DesiredWeight": row["DesiredWeight"],  
                "DeltaWeight": -row["DesiredWeight"]  
            })
        return pd.DataFrame(rows)  

    sheep_sorted = sheep_df_part.sort_values("Weight_kg", ascending=False).reset_index(drop=True)  # Sort sheep by weight (heaviest first) for greedy assignment

    owners = [d["Owner"] for d in desired_by_owner]  
    desired = {d["Owner"]: float(d["DesiredWeight"]) for d in desired_by_owner}  
    allocated = {o: 0.0 for o in owners}  # Track cumulative allocated weight per owner
    owned_sheep = {o: [] for o in owners}  # Track the list of SheepIDs assigned to each owner

    for _, row in sheep_sorted.iterrows():  # Iterate heaviest to lightest sheep
        remaining = {o: desired[o] - allocated[o] for o in owners}  # Compute remaining need per owner
        # Choose owner with largest remaining need; tie-breaker: owner with smallest currently allocated weight
        best_owner = sorted(owners, key=lambda o: (-(remaining[o]), allocated[o]))[0]  # Sort by (-remaining, allocated) and pick first
        owned_sheep[best_owner].append(row["SheepID"])  
        allocated[best_owner] += float(row["Weight_kg"]) 

    out_rows = []  # Prepare output rows
    for o in owners:  # For each owner, write a summary row
        dw = desired[o]  
        aw = allocated[o]  
        out_rows.append({  
            "PartID": part_id, 
            "Owner": o,  
            "SheepIDs": ", ".join(owned_sheep[o]),  
            "AllocatedWeight": round(aw, 2),
            "DesiredWeight": round(dw, 2),
            "DeltaWeight": round(aw - dw, 2)  
        })
    return pd.DataFrame(out_rows)  


def run_allocation(df_sheep_in, df_settings_in, df_owners_in):
    SKINN_PER_MARK = 240  # Constants (unit system)
    SKINN_PER_GYLLIN = 16

    try:
        n_parts = int(
            df_settings_in.loc[
                df_settings_in["Setting"] == "NumberOfParts", "Value"
            ].values[0]
        )
    except Exception:
        n_parts = 8

    sep_by_sex_str = str(
        df_settings_in.loc[
            df_settings_in["Setting"] == "SeparateBySex", "Value"
        ].values[0]
    ).upper()
    separate_by_sex = sep_by_sex_str in ["TRUE", "YES", "1", "Y"]

    df_owners = df_owners_in.copy()

    for col in ["mark", "gyllin", "skinn"]:
        if col not in df_owners.columns:
            df_owners[col] = 0
        df_owners[col] = pd.to_numeric(df_owners[col], errors="coerce").fillna(0)

    df_owners["Skinn"] = (
        df_owners["mark"] * SKINN_PER_MARK
        + df_owners["gyllin"] * SKINN_PER_GYLLIN
        + df_owners["skinn"]
    )

    # Normalize ownership strictly within each part
    skinn_by_part = df_owners.groupby("PartID")["Skinn"].transform("sum")
    df_owners["SharePct"] = np.where(
        skinn_by_part > 0,
        df_owners["Skinn"] / skinn_by_part * 100,
        0.0,
    )

    df = df_sheep_in.copy()
    df["Weight_kg"] = pd.to_numeric(df["Weight_kg"], errors="coerce").fillna(0.0)
    df["Sex"] = (
        df["Sex"]
        .astype(str)
        .str.upper()
        .str[0]
        .map({"M": "M", "F": "F"})
        .fillna("U")
    )

    assignments = []

    if separate_by_sex:
        for sex in ["M", "F"]:
            sub = df[df["Sex"] == sex].reset_index(drop=True)
            bins = lpt_partition(list(sub["Weight_kg"]), n_parts)
            for i, b in enumerate(bins):
                assignments.append((sub.loc[i, "SheepID"], b + 1))
    else:
        bins = lpt_partition(list(df["Weight_kg"]), n_parts)
        for i, b in enumerate(bins):
            assignments.append((df.loc[i, "SheepID"], b + 1))

    df_assign = pd.DataFrame(assignments, columns=["SheepID", "PartID"])
    df_out = df.merge(df_assign, on="SheepID", how="left")

    # Part summaries
    grp = (
        df_out.groupby(["PartID", "Sex"])["Weight_kg"]
        .sum()
        .reset_index()
    )
    pivot = grp.pivot(index="PartID", columns="Sex", values="Weight_kg").fillna(0.0)
    for col in ["M", "F", "U"]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot["TotalWeight_kg"] = pivot[["M", "F", "U"]].sum(axis=1)
    df_part_summary = pivot.reset_index().sort_values("PartID")

    # Desired owner weights (LOCAL)
    owner_desired = []
    for _, row in df_owners.iterrows():
        part = int(row["PartID"])
        share = row["SharePct"] / 100.0
        total_weight = df_part_summary.loc[
            df_part_summary["PartID"] == part, "TotalWeight_kg"
        ]
        total_weight = float(total_weight.values[0]) if len(total_weight) else 0.0

        owner_desired.append({
            "PartID": part,
            "Owner": row["Owner"],
            "SharePct": round(row["SharePct"], 4),
            "DesiredWeight_kg": round(share * total_weight, 3),
        })

    df_owner_desired = pd.DataFrame(owner_desired).sort_values(["PartID", "Owner"])

    # Whole-sheep allocation
    alloc_all = []
    for part in range(1, n_parts + 1):
        sheep_part = df_out[df_out["PartID"] == part][["SheepID", "Weight_kg"]]
        desired_part = df_owner_desired[df_owner_desired["PartID"] == part]

        desired_list = [
            {"Owner": r["Owner"], "DesiredWeight": r["DesiredWeight_kg"]}
            for _, r in desired_part.iterrows()
        ]

        alloc_all.append(
            allocate_whole_sheep_to_owners(part, sheep_part, desired_list)
        )

    df_whole_alloc = (
        pd.concat(alloc_all, ignore_index=True)
        if alloc_all else pd.DataFrame()
    )

    return (
        df_out,
        df_part_summary,
        df_owners[["PartID", "Owner", "Skinn", "SharePct"]],
        df_owner_desired,
        df_whole_alloc,
    )

def main(xlsx_path): 
    xlsx = Path(xlsx_path)  
    if not xlsx.exists(): 
        print(f"File not found: {xlsx}") 
        sys.exit(1)  

    # Read required sheets from the workbook in one pass
    with pd.ExcelFile(xlsx) as xf:  
        df_sheep = pd.read_excel(xf, "Sheep") 
        df_owners = pd.read_excel(xf, "Owners")  
        df_settings = pd.read_excel(xf, "Settings")  

    # Run the allocation engine to compute outputs from the inputs
    df_parted, df_part_summary, df_owners_calc, df_owner_desired, df_whole_alloc = run_allocation(
        df_sheep, df_settings, df_owners  # Pass the 3 input tables into the core function
    )

    # Write computed tables back into the same workbook, replacing existing sheets if they exist
    with pd.ExcelWriter(xlsx, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:  
        df_parted.to_excel(writer, sheet_name="Part_Assignments", index=False)  
        df_part_summary.to_excel(writer, sheet_name="Part_Summary", index=False) 
        df_owners_calc[["PartID", "Owner", "SharePct"]].sort_values(["PartID", "Owner"]).to_excel(
            writer, sheet_name="Owner_Shares", index=False  
        )
        df_owner_desired.to_excel(writer, sheet_name="Owner_Desired", index=False)  
        df_whole_alloc.to_excel(writer, sheet_name="Whole_Sheep_Allocation", index=False)  

    print("OK: Allocation updated.")  

if __name__ == "__main__":  
    if len(sys.argv) < 2:  
        print("Usage: python3 sheep_divider_annotated.py sheep_allocation_template3.xlsx")  # Print correct usage instructions
        sys.exit(1)  
    main(sys.argv[1])  
