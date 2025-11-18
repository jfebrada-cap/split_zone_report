#!/usr/bin/env python3

import os
import glob
import re
import pandas as pd
import chardet
import numpy as np
from datetime import datetime

# -------------------------------
# Utility Functions
# -------------------------------

def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read(50000))
    return result["encoding"] or "utf-8"

def detect_delimiter(file_path, encoding):
    with open(file_path, "r", encoding=encoding, errors="ignore") as f:
        sample = f.read(4096)
    delimiters = [",", ";", "\t", "|"]
    counts = {d: sample.count(d) for d in delimiters}
    return max(counts, key=counts.get)

def get_latest_csv():
    csv_files = sorted(glob.glob("raw/*.csv"), key=os.path.getmtime, reverse=True)
    if not csv_files:
        raise FileNotFoundError("No CSV files found in 'raw' directory.")
    return csv_files[0]

# -------------------------------
# Core Functions
# -------------------------------

def extract_all_tag_values(tag_str):
    if pd.isna(tag_str):
        return {}
    tag_dict = {}
    tag_pairs = str(tag_str).split(';')
    for pair in tag_pairs:
        if '=' in pair:
            key, value = pair.split('=', 1)
            tag_dict[key.strip()] = value.strip()
    return tag_dict

def extract_tag_value(tag_str, key):
    if pd.isna(tag_str):
        return None
    match = re.search(rf"{key}=([^;]+)", str(tag_str))
    return match.group(1).strip() if match else None

def classify_zone(zone_id):
    if pd.isna(zone_id):
        return "Unknown"
    zone_str = str(zone_id).lower()
    if "sgp1a" in zone_str or "ap-southeast-1a" in zone_str:
        return "Zone A"
    elif "sgp1b" in zone_str or "ap-southeast-1b" in zone_str:
        return "Zone B"
    else:
        if zone_str.endswith('a'):
            return "Zone A"
        elif zone_str.endswith('b'):
            return "Zone B"
        else:
            return "Unknown"

def normalize_identifier(identifier):
    if pd.isna(identifier):
        return "Unknown"
    identifier_str = str(identifier)
    normalized = re.sub(r'-\d+(-\d+)*$', '', identifier_str)
    normalized = re.sub(r'\d+$', '', normalized)
    return normalized

def load_file():
    file_path = get_latest_csv()
    print(f"Loading {os.path.basename(file_path)} ...")
    enc = detect_encoding(file_path)
    print(f"   • Detected encoding: {enc}")
    delim = detect_delimiter(file_path, enc)
    print(f"   • Using delimiter: '{delim}'")
    df = pd.read_csv(file_path, encoding=enc, delimiter=delim, engine="python")
    print(f"   • Loaded {len(df):,} rows × {len(df.columns)} columns")
    df.columns = df.columns.str.strip()

    if "Identifier_Name" in df.columns:
        df.rename(columns={"Identifier_Name": "identifier"}, inplace=True)
    elif "identifier" not in df.columns:
        possible = [c for c in df.columns if "id" in c.lower() or "identifier" in c.lower()]
        if possible:
            df.rename(columns={possible[0]: "identifier"}, inplace=True)
            print(f"   • Mapped '{possible[0]}' → 'identifier'")
        else:
            raise KeyError("Missing required column: 'identifier'")

    # Extract component from Tags
    if "Tags" in df.columns:
        df["component"] = df["Tags"].apply(
            lambda x: extract_tag_value(x, "Application Component") or
                      extract_tag_value(x, "component") or
                      extract_tag_value(x, "Name") or None
        )
        all_tags = set()
        for tags in df["Tags"].dropna():
            all_tags.update(extract_all_tag_values(tags).keys())
        for tag_key in all_tags:
            df[f"tag_{tag_key}"] = df["Tags"].apply(lambda x: extract_tag_value(x, tag_key))
    else:
        df["component"] = None

    df["component_group"] = df.apply(
        lambda row: row["component"] if pd.notna(row["component"]) and row["component"] not in ["", "Unknown", "unknown"]
                    else normalize_identifier(row["identifier"]),
        axis=1
    )

    df["AZ_Class"] = df["Zone_Id"].apply(classify_zone) if "Zone_Id" in df.columns else "Unknown"

    # Convert metrics to numeric
    metric_cols = ["Max_CPU_Util","P95_CPU","Max_Mem_Util","P95_Mem","P95_Disk",
                   "avg_CPU_Util","avg_Mem_Util","vCPUs","Memory_GB","diskusage_util"]
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# -------------------------------
# Production Filter (Modified)
# -------------------------------

def filter_production(df: pd.DataFrame) -> pd.DataFrame:
    env_cols = [c for c in df.columns if c.lower() in ["environment", "env", "stage"]]
    
    # Create masks for different conditions
    prod_mask = pd.Series(False, index=df.index)
    blank_env_mask = pd.Series(False, index=df.index)
    non_prod_mask = pd.Series(False, index=df.index)
    
    # Define non-production environment patterns to exclude
    non_prod_patterns = [r"^Dev", r"development", r"^staging", r"^stage", r"^test", 
                         r"^qa", r"^uat", r"^preprod", r"^sandbox", r"^demo", r"^ctu",
                         r"^k8s", r"^nonprod", r"^perf", r"^zoloz", r"^tools", r"^security",r"^sit"]
    
    # Check environment columns
    for col in env_cols:
        col_series = df[col].astype(str).str.lower()
        
        # Production patterns
        prod_mask |= col_series.str.contains(r"^prod|production$", na=False)
        
        # Blank environment patterns
        blank_env_mask |= (col_series.isin(["", "nan", "none", "null"]) | df[col].isna())
        
        # Non-production patterns to exclude
        for pattern in non_prod_patterns:
            non_prod_mask |= col_series.str.contains(pattern, na=False)
    
    # Check Tags for environment values
    if "Tags" in df.columns:
        # Production tags
        tags_prod_mask = df["Tags"].apply(
            lambda x: any(v.lower() in ["prod", "production"] 
                         for v in extract_all_tag_values(x).values())
        )
        prod_mask |= tags_prod_mask
        
        # Blank environment tags
        tags_env_blank_mask = df["Tags"].apply(
            lambda x: extract_tag_value(x, "Environment") in [None, "", "nan", "none"] 
                     if pd.notna(x) else True
        )
        blank_env_mask |= tags_env_blank_mask
        
        # Non-production tags
        tags_non_prod_mask = df["Tags"].apply(
            lambda x: any(v.lower() in ["dev", "development", "staging", "stage", "test", 
                                       "qa", "uat", "preprod", "sandbox", "demo"]
                         for v in extract_all_tag_values(x).values())
        )
        non_prod_mask |= tags_non_prod_mask
    
    # Include: (Production OR Blank environment) AND NOT Non-production
    final_mask = (prod_mask | blank_env_mask) & ~non_prod_mask
    
    filtered_df = df[final_mask].copy()
    print(f"Filtered production environment rows: {len(filtered_df)} / {len(df)}")
    
    # Print breakdown for transparency
    print(f"   • Explicit production rows: {prod_mask.sum()}")
    print(f"   • Blank environment rows: {blank_env_mask.sum()}")
    print(f"   • Non-production rows excluded: {non_prod_mask.sum()}")
    print(f"   • Final production rows: {len(filtered_df)}")
    
    return filtered_df

# -------------------------------
# Critical Applications List
# -------------------------------

CRITICAL_APPLICATIONS = {
    "apacquireprod", "apbizprod", "apcashier", "apfundprod", "apmobileappng",
    "apmobileprod", "gcfund", "gcrouter", "gcuser", "ifcassetflux",
    "ifccardcenter", "ifcdart", "ifcdatabus", "ifcfluxbatch", "ifcgotone",
    "ifcinnertrans", "ifclimitcenter", "ifcriskcloud", "ifcsupergw",
    "mpaas_mdap", "mpaasgw"
}

def is_critical_application(app_name):
    """Check if the application is in the critical applications list"""
    if pd.isna(app_name):
        return ""
    app_str = str(app_name).strip().lower()
    return "Critical-app" if app_str in CRITICAL_APPLICATIONS else ""

# -------------------------------
# Consolidation
# -------------------------------

def safe_mode(series):
    s = series.dropna()
    return s.mode().iloc[0] if not s.empty else None

def compute_p95(series):
    s = series.dropna()
    return np.percentile(s, 95) if not s.empty else 0

def consolidate(df: pd.DataFrame) -> pd.DataFrame:
    def agg_group(g):
        result = {}
        total_instances = len(g)
        total_vcpus = g["vCPUs"].sum() if "vCPUs" in g.columns else total_instances
        total_mem = g["Memory_GB"].sum() if "Memory_GB" in g.columns else total_instances

        result["Total vCPUs"] = total_vcpus
        result["Total Memory"] = total_mem
        result["Total of Instances"] = total_instances

        az_a = g[g["AZ_Class"]=="Zone A"]
        az_b = g[g["AZ_Class"]=="Zone B"]
        result["Zone A Instance #"] = len(az_a)
        result["Zone B Instance #"] = len(az_b)

        def az_metrics(subset, label):
            metrics = {}
            if len(subset) == 0:
                metrics[f"{label} Max CPU/P95"] = 0
                metrics[f"{label} Max Mem/P95"] = 0
                metrics[f"{label} Disk Util/P95"] = 0
                return metrics
            metrics[f"{label} Max CPU/P95"] = compute_p95(subset["P95_CPU"]) if "P95_CPU" in subset.columns else subset["Max_CPU_Util"].max(skipna=True)
            metrics[f"{label} Max Mem/P95"] = compute_p95(subset["P95_Mem"]) if "P95_Mem" in subset.columns else subset["Max_Mem_Util"].max(skipna=True)
            metrics[f"{label} Disk Util/P95"] = compute_p95(subset["P95_Disk"]) if "P95_Disk" in subset.columns else subset["diskusage_util"].mean(skipna=True) if "diskusage_util" in subset.columns else 0
            return metrics

        result.update(az_metrics(az_a, "Zone A"))
        result.update(az_metrics(az_b, "Zone B"))

        result["Max CPU Util"] = g["Max_CPU_Util"].max(skipna=True) if "Max_CPU_Util" in g.columns else 0
        result["P95 CPU"] = compute_p95(g["P95_CPU"]) if "P95_CPU" in g.columns else 0
        result["Max Mem Util"] = g["Max_Mem_Util"].max(skipna=True) if "Max_Mem_Util" in g.columns else 0
        result["P95 Mem"] = compute_p95(g["P95_Mem"]) if "P95_Mem" in g.columns else 0
        result["Diskutil"] = g["diskusage_util"].mean(skipna=True) if "diskusage_util" in g.columns else 0
        result["P95 Disk"] = compute_p95(g["P95_Disk"]) if "P95_Disk" in g.columns else 0

        result["Tribe"] = safe_mode(g["tag_tribe"]) if "tag_tribe" in g.columns else None
        result["Platform"] = safe_mode(g["tag_platform"]) if "tag_platform" in g.columns else None
        result["Validation"] = None

        env_cols = [c for c in g.columns if c.lower() in ["environment","env","stage"]]
        env_val = None
        for col in env_cols:
            val = safe_mode(g[col])
            if val is not None:
                env_val = val
                break
        if env_val is None and "Tags" in g.columns:
            env_val = safe_mode(g["Tags"].apply(lambda x: extract_tag_value(x,"Environment")))
        result["Environment"] = env_val

        return pd.Series(result)

    grouped = df.groupby("component_group", group_keys=False).apply(agg_group).reset_index()
    grouped.rename(columns={"component_group":"Application Baseline"}, inplace=True)
    
    # Add Critical column after Application Baseline
    grouped["Critical"] = grouped["Application Baseline"].apply(is_critical_application)
    
    for col in grouped.select_dtypes(include=np.number).columns:
        grouped[col] = grouped[col].round(2)
    
    return grouped

# -------------------------------
# Save and Highlight
# -------------------------------

def save_output(df):
    import xlsxwriter
    output_dir = "transformed"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    final_columns = [
        "Application Baseline","Critical","Environment","Total of Instances","Total vCPUs","Total Memory",
        "Max CPU Util","P95 CPU","Max Mem Util","P95 Mem","Diskutil","P95 Disk",
        "Zone A Instance #","Zone B Instance #","Zone A Max CPU/P95","ZoneB Max CPU/P95",
        "Zone A Max Mem/P95","Zone B Max Mem/P95","Zone A Disk Util/P95","Zone B Disk Util/P95",
        "Tribe","Platform","Validation"
    ]
    final_columns = [c for c in final_columns if c in df.columns]

    out_path = os.path.join(output_dir, f"ecs_utilization_summary_{timestamp}.xlsx")
    with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
        df[final_columns].to_excel(writer, index=False, sheet_name="Summary")
        workbook  = writer.book
        worksheet = writer.sheets["Summary"]

        # Conditional formatting
        fmt_orange = workbook.add_format({'bg_color': '#FFA500'})
        fmt_red = workbook.add_format({'bg_color': '#FF0000'})

        highlight_cols = [
            "Max CPU Util","P95 CPU","Max Mem Util","P95 Mem",
            "Diskutil","P95 Disk",
            "Zone A Max CPU/P95","ZoneB Max CPU/P95",
            "Zone A Max Mem/P95","Zone B Max Mem/P95",
            "Zone A Disk Util/P95","Zone B Disk Util/P95"
        ]
        for col in highlight_cols:
            if col in df.columns:
                col_idx = final_columns.index(col)
                worksheet.conditional_format(1, col_idx, len(df), col_idx,
                                             {'type':'cell', 'criteria': 'between', 'minimum':75, 'maximum':89, 'format':fmt_orange})
                worksheet.conditional_format(1, col_idx, len(df), col_idx,
                                             {'type':'cell', 'criteria': 'between', 'minimum':90, 'maximum':100, 'format':fmt_red})

    latest_path = os.path.join(output_dir, "ecs_utilization_summary_latest.xlsx")
    df[final_columns].to_excel(latest_path, index=False)
    print(f"Saved consolidated output: {out_path} (backup: {latest_path})")

# -------------------------------
# Main
# -------------------------------

def main():
    print("Starting ECS Utilization Summary with Zone A/B Split...\n")
    df = load_file()
    df_prod = filter_production(df)
    consolidated = consolidate(df_prod)
    save_output(consolidated)
    print("\n Done — AZ capacity summary with consolidated metrics complete!")

if __name__ == "__main__":
    main()
