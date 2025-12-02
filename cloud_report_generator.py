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

def get_resource_files():
    """Get all resource files from raw directory"""
    resource_files = {}
    
    # ECS files
    ecs_files = sorted(glob.glob("raw/ecs_*.csv"), key=os.path.getmtime, reverse=True)
    if ecs_files:
        resource_files["ECS"] = ecs_files[0]
    
    # RDS files
    rds_files = sorted(glob.glob("raw/rds_*.csv"), key=os.path.getmtime, reverse=True)
    if rds_files:
        resource_files["RDS"] = rds_files[0]
    
    # Cache files
    cache_files = sorted(glob.glob("raw/cache_*.csv"), key=os.path.getmtime, reverse=True)
    if cache_files:
        resource_files["Cache"] = cache_files[0]
    
    return resource_files

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

def load_resource_file(file_path, resource_type):
    """Load and process resource file based on type"""
    print(f"Loading {os.path.basename(file_path)} ({resource_type}) ...")
    enc = detect_encoding(file_path)
    print(f"   • Detected encoding: {enc}")
    delim = detect_delimiter(file_path, enc)
    print(f"   • Using delimiter: '{delim}'")
    df = pd.read_csv(file_path, encoding=enc, delimiter=delim, engine="python")
    print(f"   • Loaded {len(df):,} rows × {len(df.columns)} columns")
    df.columns = df.columns.str.strip()
    
    # Add resource type column
    df["Resource_Type"] = resource_type
    
    # Different processing based on resource type
    if resource_type == "ECS":
        return process_ecs_file(df, file_path)
    elif resource_type == "RDS":
        return process_rds_file(df, file_path)
    elif resource_type == "Cache":
        return process_cache_file(df, file_path)
    else:
        return df

def process_ecs_file(df, file_path):
    """Process ECS specific file"""
    # Extract date from filename for datetime column
    filename = os.path.basename(file_path)
    date_match = re.search(r'ecs_(\d{8})\d+\.csv', filename)
    file_date = ""
    if date_match:
        date_str = date_match.group(1)
        file_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    
    if "Identifier_Name" in df.columns:
        df.rename(columns={"Identifier_Name": "identifier"}, inplace=True)
    elif "identifier" not in df.columns:
        possible = [c for c in df.columns if "id" in c.lower() or "identifier" in c.lower()]
        if possible:
            df.rename(columns={possible[0]: "identifier"}, inplace=True)
            print(f"   • Mapped '{possible[0]}' → 'identifier'")
        else:
            raise KeyError("Missing required column: 'identifier'")

    # Extract all tags from Tags column
    if "Tags" in df.columns:
        # Extract component from Tags
        df["component"] = df["Tags"].apply(
            lambda x: extract_tag_value(x, "Application Component") or
                      extract_tag_value(x, "component") or
                      extract_tag_value(x, "Name") or None
        )
        
        # Extract other tags
        tag_keys = ["project", "environment", "environment_group", 
                    "tribe", "squad", "platform", "alerting"]
        
        for tag_key in tag_keys:
            df[tag_key] = df["Tags"].apply(lambda x: extract_tag_value(x, tag_key))
        
        # Also extract any other tags for reference
        all_tags = set()
        for tags in df["Tags"].dropna():
            all_tags.update(extract_all_tag_values(tags).keys())
        for tag_key in all_tags:
            if tag_key not in tag_keys + ["component", "Application Component", "Name"]:
                df[f"tag_{tag_key}"] = df["Tags"].apply(lambda x: extract_tag_value(x, tag_key))
    else:
        df["component"] = None
        # Initialize other tag columns
        tag_keys = ["project", "environment", "environment_group", 
                    "tribe", "squad", "platform", "alerting"]
        for tag_key in tag_keys:
            df[tag_key] = None

    df["component_group"] = df.apply(
        lambda row: row["component"] if pd.notna(row["component"]) and row["component"] not in ["", "Unknown", "unknown"]
                    else normalize_identifier(row["identifier"]),
        axis=1
    )

    df["AZ_Class"] = df["Zone_Id"].apply(classify_zone) if "Zone_Id" in df.columns else "Unknown"
    
    # Add datetime from filename
    df["datetime"] = file_date

    # Ensure platform column exists - leave empty if not found
    if "platform" not in df.columns:
        df["platform"] = ""
    
    # Map platform to standard categories (DAAS, PAAS, SAAS, OTHER)
    # If platform is empty, categorize as OTHER
    def map_platform(platform):
        if pd.isna(platform) or platform == "":
            return "OTHER"
        platform_str = str(platform).upper()
        if "DAAS" in platform_str:
            return "DAAS"
        elif "PAAS" in platform_str:
            return "PAAS"
        elif "SAAS" in platform_str:
            return "SAAS"
        else:
            return "OTHER"
    
    df["Platform_Category"] = df["platform"].apply(map_platform)
    
    # Convert metrics to numeric
    metric_cols = ["Max_CPU_Util","P95_CPU","Max_Mem_Util","P95_Mem","P95_Disk",
                   "avg_CPU_Util","avg_Mem_Util","vCPUs","Memory_GB","diskusage_util"]
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df

def process_rds_file(df, file_path):
    """Process RDS specific file"""
    # Extract date from filename for datetime column
    filename = os.path.basename(file_path)
    date_match = re.search(r'rds_(\d{8})\d+\.csv', filename)
    file_date = ""
    if date_match:
        date_str = date_match.group(1)
        file_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    
    # Add datetime from filename
    df["datetime"] = file_date
    
    # Rename Instance_Type to Instance_Class if it exists
    if "Instance_Type" in df.columns:
        df.rename(columns={"Instance_Type": "Instance_Class"}, inplace=True)
        print(f"   • Renamed 'Instance_Type' → 'Instance_Class'")
    
    # Rename columns to match expected format
    column_mapping = {
        "Max_CPU_Util": "RDS_Max_CPU_Util",
        "P95_CPU": "RDS_P95_CPU",
        "Max_Mem_Util": "RDS_Max_Mem_Util",
        "P95_Mem": "RDS_P95_Mem",
        "DISK_Usage": "RDS_DISK_Usage"
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)
    
    # Extract platform from tags or platform column
    if "Tags" in df.columns:
        df["platform"] = df["Tags"].apply(lambda x: extract_tag_value(x, "platform"))
    elif "platform" in df.columns:
        df["platform"] = df["platform"]
    
    # Ensure platform column exists - leave empty if not found
    if "platform" not in df.columns:
        df["platform"] = ""
    
    # Map platform to standard categories
    # If platform is empty, categorize as OTHER
    def map_platform(platform):
        if pd.isna(platform) or platform == "":
            return "OTHER"
        platform_str = str(platform).upper()
        if "DAAS" in platform_str:
            return "DAAS"
        elif "PAAS" in platform_str:
            return "PAAS"
        elif "SAAS" in platform_str:
            return "SAAS"
        else:
            return "OTHER"
    
    df["Platform_Category"] = df["platform"].apply(map_platform)
    
    # Convert RDS metrics to numeric
    rds_metric_cols = [
        "RDS_Max_CPU_Util", "RDS_P95_CPU", "RDS_Max_Mem_Util", 
        "RDS_P95_Mem", "RDS_DISK_Usage", "vCPUs", "Memory_GB",
        "Max_supported_connections"
    ]
    
    for col in rds_metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df

def process_cache_file(df, file_path):
    """Process Cache specific file"""
    # Extract date from filename for datetime column
    filename = os.path.basename(file_path)
    date_match = re.search(r'cache_(\d{8})\d+\.csv', filename)
    file_date = ""
    if date_match:
        date_str = date_match.group(1)
        file_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    
    # Add datetime from filename
    df["datetime"] = file_date
    
    # Rename columns to match expected format
    column_mapping = {
        "Max_CPU_Util": "Cache_Max_CPU_Util",
        "P95_CPU_Util": "Cache_P95_CPU_Util",
        "Max_Mem_Util": "Cache_Max_Mem_Util",
        "P95_Mem_Util": "Cache_P95_Mem_Util"
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)
    
    # Extract platform from tags or platform column
    if "Tags" in df.columns:
        df["platform"] = df["Tags"].apply(lambda x: extract_tag_value(x, "platform"))
    elif "platform" in df.columns:
        df["platform"] = df["platform"]
    
    # Ensure platform column exists - leave empty if not found
    if "platform" not in df.columns:
        df["platform"] = ""
    
    # Map platform to standard categories
    # If platform is empty, categorize as OTHER
    def map_platform(platform):
        if pd.isna(platform) or platform == "":
            return "OTHER"
        platform_str = str(platform).upper()
        if "DAAS" in platform_str:
            return "DAAS"
        elif "PAAS" in platform_str:
            return "PAAS"
        elif "SAAS" in platform_str:
            return "SAAS"
        else:
            return "OTHER"
    
    df["Platform_Category"] = df["platform"].apply(map_platform)
    
    # Convert Cache metrics to numeric
    cache_metric_cols = [
        "Cache_Max_CPU_Util", "Cache_P95_CPU_Util", 
        "Cache_Max_Mem_Util", "Cache_P95_Mem_Util",
        "Memory_GiB", "Instance_Specs_Storage_GiB", 
        "Max_Supported_Connections", "Max_Connections", "P95_Connections"
    ]
    
    for col in cache_metric_cols:
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
        blank_env_mask |= (col_series.isin(["", "OTHER", "none", "null"]) | df[col].isna())
        
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
            lambda x: extract_tag_value(x, "Environment") in [None, "", "OTHER", "none"] 
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
# Consolidation (for ECS only)
# -------------------------------

def safe_mode(series):
    s = series.dropna()
    return s.mode().iloc[0] if not s.empty else None

def compute_p95(series):
    s = series.dropna()
    return np.percentile(s, 95) if not s.empty else 0

def get_application_metric(zone_a_value, zone_b_value):
    """
    Get application-level metric by taking the maximum between Zone A and Zone B.
    If one zone is 0 (no instances), use the other zone's value.
    """
    if zone_a_value == 0 and zone_b_value == 0:
        return 0
    elif zone_a_value == 0:
        return zone_b_value
    elif zone_b_value == 0:
        return zone_a_value
    else:
        return max(zone_a_value, zone_b_value)

def consolidate_ecs(df: pd.DataFrame) -> pd.DataFrame:
    def agg_group(g):
        result = {}
        
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

        # Get individual zone metrics
        zone_a_metrics = az_metrics(az_a, "Zone A")
        zone_b_metrics = az_metrics(az_b, "Zone B")
        
        # Calculate application-level metrics (maximum of Zone A and Zone B)
        result["Application CPU/P95"] = get_application_metric(
            zone_a_metrics.get("Zone A Max CPU/P95", 0),
            zone_b_metrics.get("Zone B Max CPU/P95", 0)
        )
        
        result["Application Mem/P95"] = get_application_metric(
            zone_a_metrics.get("Zone A Max Mem/P95", 0),
            zone_b_metrics.get("Zone B Max Mem/P95", 0)
        )
        
        result["Application Disk Util/P95"] = get_application_metric(
            zone_a_metrics.get("Zone A Disk Util/P95", 0),
            zone_b_metrics.get("Zone B Disk Util/P95", 0)
        )

        # Get platform from the group
        result["Platform_Category"] = safe_mode(g["Platform_Category"]) if "Platform_Category" in g.columns else "OTHER"
        
        # Get specific tag values from the group
        tag_columns = ["datetime", "environment", "environment_group", 
                      "tribe", "squad", "platform", "alerting"]
        
        for tag_col in tag_columns:
            if tag_col in g.columns:
                result[tag_col] = safe_mode(g[tag_col])

        return pd.Series(result)

    # Fix for FutureWarning: use reset_index() differently
    grouped = df.groupby("component_group").apply(agg_group).reset_index()
    grouped.rename(columns={"component_group":"Application Baseline"}, inplace=True)
    
    # Add Critical column after Application Baseline
    grouped["Critical"] = grouped["Application Baseline"].apply(is_critical_application)
    
    for col in grouped.select_dtypes(include=np.number).columns:
        grouped[col] = grouped[col].round(2)
    
    return grouped

def get_resource_metrics(df, resource_type):
    """Get aggregated metrics for RDS/Cache resources"""
    result = {
        "Total Instances": len(df),
        "CPU/P95": 0,
        "Mem/P95": 0,
        "Disk Util/P95": 0
    }
    
    if len(df) == 0:
        return result
    
    # Get CPU/P95 based on resource type
    if resource_type == "RDS":
        if "RDS_P95_CPU" in df.columns:
            result["CPU/P95"] = compute_p95(df["RDS_P95_CPU"])
        elif "RDS_Max_CPU_Util" in df.columns:
            result["CPU/P95"] = compute_p95(df["RDS_Max_CPU_Util"])
        
        # Get Mem/P95 for RDS
        if "RDS_P95_Mem" in df.columns:
            result["Mem/P95"] = compute_p95(df["RDS_P95_Mem"])
        elif "RDS_Max_Mem_Util" in df.columns:
            result["Mem/P95"] = compute_p95(df["RDS_Max_Mem_Util"])
        
        # Get Disk Util/P95 for RDS
        if "RDS_DISK_Usage" in df.columns:
            result["Disk Util/P95"] = compute_p95(df["RDS_DISK_Usage"])
    
    elif resource_type == "Cache":
        if "Cache_P95_CPU_Util" in df.columns:
            result["CPU/P95"] = compute_p95(df["Cache_P95_CPU_Util"])
        elif "Cache_Max_CPU_Util" in df.columns:
            result["CPU/P95"] = compute_p95(df["Cache_Max_CPU_Util"])
        
        # Get Mem/P95 for Cache
        if "Cache_P95_Mem_Util" in df.columns:
            result["Mem/P95"] = compute_p95(df["Cache_P95_Mem_Util"])
        elif "Cache_Max_Mem_Util" in df.columns:
            result["Mem/P95"] = compute_p95(df["Cache_Max_Mem_Util"])
        
        # Cache doesn't have disk metrics, so set to 0
        result["Disk Util/P95"] = 0
    
    # Round the metrics
    for key in ["CPU/P95", "Mem/P95", "Disk Util/P95"]:
        result[key] = round(result[key], 2)
    
    return result

def create_executive_dashboard(all_data, raw_ecs_data=None, raw_rds_data=None, raw_cache_data=None):
    """Create executive dashboard-style summary for all resource types"""
    summary_data = []
    
    # Define platform categories
    platform_categories = ["DAAS", "PAAS", "SAAS", "OTHER"]
    
    # Get current date for dashboard header
    current_date = datetime.now().strftime("%m/%d/%Y")
    
    # -------------------------------
    # EXECUTIVE DASHBOARD HEADER
    # -------------------------------
    summary_data.append({"A": "A+ CLOUD RESOURCES EXECUTIVE DASHBOARD", "B": "", "C": "", "D": "", "E": "", 
                         "F": "", "G": "", "H": "", "I": "", "J": "", "K": "", "L": "", "M": ""})
    summary_data.append({"A": f"Report Date: {current_date}", "B": "", "C": "", "D": "", "E": "", 
                         "F": "", "G": "", "H": "", "I": "", "J": "", "K": "", "L": "", "M": ""})
    summary_data.append({"A": "", "B": "", "C": "", "D": "", "E": "", 
                         "F": "", "G": "", "H": "", "I": "", "J": "", "K": "", "L": "", "M": ""})
    
    for resource_type in ["ECS", "RDS", "Cache"]:
        if resource_type not in all_data:
            continue
            
        # Get the appropriate data for analysis
        # For ECS, use the consolidated data (transformed data)
        # For RDS and Cache, use the raw data as they don't get consolidated
        if resource_type == "ECS":
            data_for_analysis = all_data[resource_type]  # Use consolidated data
        elif resource_type == "RDS" and raw_rds_data is not None:
            data_for_analysis = raw_rds_data
        elif resource_type == "Cache" and raw_cache_data is not None:
            data_for_analysis = raw_cache_data
        else:
            data_for_analysis = all_data[resource_type]
        
        if data_for_analysis is None or len(data_for_analysis) == 0:
            continue
        
        # -------------------------------
        # RESOURCE SECTION HEADER
        # -------------------------------
        summary_data.append({
            "A": f"🔷 {resource_type} RESOURCE OVERVIEW", "B": "", "C": "", "D": "", "E": "", 
            "F": "", "G": "", "H": "", "I": "", "J": "", 
            "K": "", "L": "", "M": ""
        })
        
        # Add sub-header row with correct utilization ranges
        summary_data.append({
            "A": "Platform", "B": "Total Deployments", 
            "C": "CPU MAX/P95", "D": "", "E": "",
            "F": "", "G": "MEM MAX/P95", "H": "", "I": "",
            "J": "", "K": "DISK USAGE/P95", "L": "", "M": ""
        })
        
        # Add utilization range headers
        summary_data.append({
            "A": "", "B": "", 
            "C": "Green 0-75%", "D": "Amber 75-89%", "E": "Red 90-100%",
            "F": "", "G": "Green 0-75%", "H": "Amber 75-89%", "I": "Red 90-100%",
            "J": "", "K": "Green 0-75%", "L": "Amber 75-89%", "M": "Red 90-100%"
        })
        
        # Define metric columns based on resource type
        if resource_type == "ECS":
            # For ECS, we use the Application-level metrics from consolidated data
            cpu_col = "Application CPU/P95"
            mem_col = "Application Mem/P95"
            disk_col = "Application Disk Util/P95"
        elif resource_type == "RDS":
            cpu_col = "RDS_Max_CPU_Util" if "RDS_Max_CPU_Util" in data_for_analysis.columns else "RDS_P95_CPU"
            mem_col = "RDS_Max_Mem_Util" if "RDS_Max_Mem_Util" in data_for_analysis.columns else "RDS_P95_Mem"
            disk_col = "RDS_DISK_Usage"
        elif resource_type == "Cache":
            cpu_col = "Cache_Max_CPU_Util" if "Cache_Max_CPU_Util" in data_for_analysis.columns else "Cache_P95_CPU_Util"
            mem_col = "Cache_Max_Mem_Util" if "Cache_Max_Mem_Util" in data_for_analysis.columns else "Cache_P95_Mem_Util"
            disk_col = None  # Cache doesn't have disk metrics
        
        # Calculate metrics for each platform
        total_deployments = 0
        platform_totals = []
        
        for platform in platform_categories:
            # Filter instances for this platform
            if "Platform_Category" in data_for_analysis.columns:
                platform_data = data_for_analysis[data_for_analysis["Platform_Category"] == platform]
            elif "platform" in data_for_analysis.columns:
                platform_data = data_for_analysis[data_for_analysis["platform"].astype(str).str.upper() == platform]
            else:
                platform_data = data_for_analysis if platform == "OTHER" else pd.DataFrame()
            
            # For ECS, count applications (after consolidation), for others count instances
            num_deployments = len(platform_data)
            total_deployments += num_deployments
            
            # Initialize counts
            cpu_green = cpu_amber = cpu_red = 0
            mem_green = mem_amber = mem_red = 0
            disk_green = disk_amber = disk_red = 0
            
            if num_deployments > 0:
                # CPU metrics with CORRECT ranges
                if cpu_col in data_for_analysis.columns:
                    cpu_values = pd.to_numeric(platform_data[cpu_col], errors='coerce').fillna(0)
                    # COUNT EACH DEPLOYMENT (application or instance)
                    cpu_green = ((cpu_values >= 0) & (cpu_values <= 75)).sum()  # Green: 0-75% inclusive
                    cpu_amber = ((cpu_values > 75) & (cpu_values <= 89)).sum()  # Amber: 75-89% inclusive
                    cpu_red = (cpu_values >= 90).sum()  # Red: 90-100% inclusive
                
                # Memory metrics with CORRECT ranges
                if mem_col in data_for_analysis.columns:
                    mem_values = pd.to_numeric(platform_data[mem_col], errors='coerce').fillna(0)
                    # COUNT EACH DEPLOYMENT (application or instance)
                    mem_green = ((mem_values >= 0) & (mem_values <= 75)).sum()  # Green: 0-75% inclusive
                    mem_amber = ((mem_values > 75) & (mem_values <= 89)).sum()  # Amber: 75-89% inclusive
                    mem_red = (mem_values >= 90).sum()  # Red: 90-100% inclusive
                
                # Disk metrics with CORRECT ranges
                if disk_col and disk_col in data_for_analysis.columns:
                    disk_values = pd.to_numeric(platform_data[disk_col], errors='coerce').fillna(0)
                    # COUNT EACH DEPLOYMENT (application or instance)
                    disk_green = ((disk_values >= 0) & (disk_values <= 75)).sum()  # Green: 0-75% inclusive
                    disk_amber = ((disk_values > 75) & (disk_values <= 89)).sum()  # Amber: 75-89% inclusive
                    disk_red = (disk_values >= 90).sum()  # Red: 90-100% inclusive
                elif resource_type == "ECS":
                    # For ECS, use Disk Util/P95 from consolidated data
                    if disk_col in data_for_analysis.columns:
                        disk_values = pd.to_numeric(platform_data[disk_col], errors='coerce').fillna(0)
                        # COUNT EACH DEPLOYMENT (application)
                        disk_green = ((disk_values >= 0) & (disk_values <= 75)).sum()  # Green: 0-75% inclusive
                        disk_amber = ((disk_values > 75) & (disk_values <= 89)).sum()  # Amber: 75-89% inclusive
                        disk_red = (disk_values >= 90).sum()  # Red: 90-100% inclusive
            
            # Format percentages with 1 decimal place for better accuracy
            cpu_green_pct = f"{cpu_green} ({cpu_green/num_deployments*100:.1f}%)" if num_deployments > 0 else "0 (0%)"
            cpu_amber_pct = f"{cpu_amber} ({cpu_amber/num_deployments*100:.1f}%)" if num_deployments > 0 else "0 (0%)"
            cpu_red_pct = f"{cpu_red} ({cpu_red/num_deployments*100:.1f}%)" if num_deployments > 0 else "0 (0%)"
            
            mem_green_pct = f"{mem_green} ({mem_green/num_deployments*100:.1f}%)" if num_deployments > 0 else "0 (0%)"
            mem_amber_pct = f"{mem_amber} ({mem_amber/num_deployments*100:.1f}%)" if num_deployments > 0 else "0 (0%)"
            mem_red_pct = f"{mem_red} ({mem_red/num_deployments*100:.1f}%)" if num_deployments > 0 else "0 (0%)"
            
            # Handle Cache disk metrics (show N/A)
            if resource_type == "Cache":
                disk_green_pct = disk_amber_pct = disk_red_pct = "N/A"
            elif disk_col and disk_col in data_for_analysis.columns:
                disk_green_pct = f"{disk_green} ({disk_green/num_deployments*100:.1f}%)" if num_deployments > 0 else "0 (0%)"
                disk_amber_pct = f"{disk_amber} ({disk_amber/num_deployments*100:.1f}%)" if num_deployments > 0 else "0 (0%)"
                disk_red_pct = f"{disk_red} ({disk_red/num_deployments*100:.1f}%)" if num_deployments > 0 else "0 (0%)"
            else:
                disk_green_pct = disk_amber_pct = disk_red_pct = "N/A"
            
            # Store for platform row
            platform_totals.append({
                "platform": platform,
                "num_deployments": num_deployments,
                "cpu_green_pct": cpu_green_pct,
                "cpu_amber_pct": cpu_amber_pct,
                "cpu_red_pct": cpu_red_pct,
                "mem_green_pct": mem_green_pct,
                "mem_amber_pct": mem_amber_pct,
                "mem_red_pct": mem_red_pct,
                "disk_green_pct": disk_green_pct,
                "disk_amber_pct": disk_amber_pct,
                "disk_red_pct": disk_red_pct
            })
        
        # Add platform rows
        for platform_data in platform_totals:
            summary_data.append({
                "A": platform_data["platform"],
                "B": platform_data["num_deployments"],
                "C": platform_data["cpu_green_pct"],
                "D": platform_data["cpu_amber_pct"],
                "E": platform_data["cpu_red_pct"],
                "F": "",
                "G": platform_data["mem_green_pct"],
                "H": platform_data["mem_amber_pct"],
                "I": platform_data["mem_red_pct"],
                "J": "",
                "K": platform_data["disk_green_pct"],
                "L": platform_data["disk_amber_pct"],
                "M": platform_data["disk_red_pct"]
            })
        
        # Add total summary row
        summary_data.append({
            "A": f"📊 {resource_type} TOTAL",
            "B": f"📍 {total_deployments}",
            "C": "", "D": "", "E": "", "F": "", 
            "G": "", "H": "", "I": "", "J": "", 
            "K": "", "L": "", "M": ""
        })
        
        # Add empty row for spacing
        summary_data.append({
            "A": "", "B": "", "C": "", "D": "", "E": "", 
            "F": "", "G": "", "H": "", "I": "", "J": "", 
            "K": "", "L": "", "M": ""
        })
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Create metrics DataFrame (for RDS/Cache aggregated metrics)
    metrics_summary = []
    for resource_type in ["RDS", "Cache"]:
        if resource_type in all_data and raw_rds_data is not None and resource_type == "RDS":
            metrics = get_resource_metrics(raw_rds_data, "RDS")
            metrics_summary.append({
                "Resource": resource_type,
                "Metric": "Total Instances",
                "Value": metrics["Total Instances"]
            })
            metrics_summary.append({
                "Resource": resource_type,
                "Metric": f"{resource_type} CPU/P95",
                "Value": metrics["CPU/P95"]
            })
            metrics_summary.append({
                "Resource": resource_type,
                "Metric": f"{resource_type} Mem/P95",
                "Value": metrics["Mem/P95"]
            })
            if resource_type == "RDS":
                metrics_summary.append({
                    "Resource": resource_type,
                    "Metric": f"{resource_type} Disk Util/P95",
                    "Value": metrics["Disk Util/P95"]
                })
        elif resource_type in all_data and raw_cache_data is not None and resource_type == "Cache":
            metrics = get_resource_metrics(raw_cache_data, "Cache")
            metrics_summary.append({
                "Resource": resource_type,
                "Metric": "Total Instances",
                "Value": metrics["Total Instances"]
            })
            metrics_summary.append({
                "Resource": resource_type,
                "Metric": f"{resource_type} CPU/P95",
                "Value": metrics["CPU/P95"]
            })
            metrics_summary.append({
                "Resource": resource_type,
                "Metric": f"{resource_type} Mem/P95",
                "Value": metrics["Mem/P95"]
            })
    
    if metrics_summary:
        metrics_df = pd.DataFrame(metrics_summary)
    else:
        metrics_df = pd.DataFrame(columns=["Resource", "Metric", "Value"])
    
    return summary_df, metrics_df

# -------------------------------
# Save and Highlight
# -------------------------------

def save_output(all_data, executive_dashboard, metrics_summary):
    import xlsxwriter
    output_dir = "transformed"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    out_path = os.path.join(output_dir, f"resource_utilization_summary_{timestamp}.xlsx")
    latest_path = os.path.join(output_dir, "resource_utilization_summary_latest.xlsx")
    
    # Enable nan_inf_to_errors option to handle NaN/INF values
    with pd.ExcelWriter(out_path, engine='xlsxwriter', engine_kwargs={'options': {'nan_inf_to_errors': True}}) as writer:
        workbook = writer.book
        
        # Define color formats for executive presentation
        fmt_green = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100', 'bold': True})  # Light green
        fmt_orange = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500', 'bold': True})  # Light orange/yellow
        fmt_red = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'bold': True})  # Light red
        
        # Executive formatting
        executive_title_format = workbook.add_format({
            'bold': True,
            'font_size': 20,
            'align': 'center',
            'valign': 'vcenter',
            'fg_color': '#1F4E78',  # Dark blue
            'font_color': 'white',
            'border': 1
        })
        
        date_format = workbook.add_format({
            'bold': True,
            'font_size': 12,
            'align': 'center',
            'valign': 'vcenter',
            'fg_color': '#4F81BD',  # Medium blue
            'font_color': 'white',
            'border': 1
        })
        
        resource_header_format = workbook.add_format({
            'bold': True,
            'font_size': 14,
            'align': 'left',
            'valign': 'vcenter',
            'fg_color': '#9DC3E6',  # Light blue
            'font_color': '#1F4E78',  # Dark blue text
            'border': 1
        })
        
        section_header_format = workbook.add_format({
            'bold': True,
            'font_size': 11,
            'align': 'center',
            'valign': 'vcenter',
            'fg_color': '#D9E1F2',  # Very light blue
            'font_color': '#1F4E78',  # Dark blue text
            'border': 1,
            'text_wrap': True
        })
        
        subheader_format = workbook.add_format({
            'bold': True,
            'font_size': 10,
            'align': 'center',
            'valign': 'vcenter',
            'fg_color': '#E2EFDA',  # Light green
            'font_color': '#385723',  # Dark green text
            'border': 1,
            'text_wrap': True
        })
        
        platform_row_format = workbook.add_format({
            'font_size': 10,
            'align': 'left',
            'valign': 'vcenter',
            'border': 1
        })
        
        total_row_format = workbook.add_format({
            'bold': True,
            'font_size': 11,
            'align': 'left',
            'valign': 'vcenter',
            'fg_color': '#FFE699',  # Light yellow
            'font_color': '#7F6000',  # Dark yellow text
            'border': 1
        })
        
        # 1. Save Executive Dashboard as first sheet
        executive_dashboard.to_excel(writer, index=False, sheet_name="Executive Dashboard", header=False)
        worksheet_dashboard = writer.sheets["Executive Dashboard"]
        
        # Apply executive formatting to the dashboard
        current_row = 0
        
        for i in range(len(executive_dashboard)):
            row_data = executive_dashboard.iloc[i]
            excel_row = current_row
            
            # Check for main title
            if "EXECUTIVE DASHBOARD" in str(row_data['A']):
                worksheet_dashboard.merge_range(f'A{excel_row+1}:M{excel_row+1}', row_data['A'], executive_title_format)
                current_row += 1
            
            # Check for date row
            elif "Report Date:" in str(row_data['A']):
                worksheet_dashboard.merge_range(f'A{excel_row+1}:M{excel_row+1}', row_data['A'], date_format)
                current_row += 1
            
            # Check for resource headers (ECS, RDS, Cache with emoji)
            elif "🔷" in str(row_data['A']):
                worksheet_dashboard.merge_range(f'A{excel_row+1}:M{excel_row+1}', row_data['A'], resource_header_format)
                current_row += 1
            
            # Check for section headers (Platform, Total Deployments, etc.)
            elif row_data['A'] == "Platform":
                # Write main header row
                for col_idx, col_letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']):
                    value = row_data[col_letter]
                    if pd.notna(value) and value != "":
                        worksheet_dashboard.write(f'{col_letter}{excel_row+1}', value, section_header_format)
                current_row += 1
            
            # Check for subheaders (Green 0-75%, etc.)
            elif row_data['C'] == "Green 0-75%":
                # Write subheader row
                for col_idx, col_letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']):
                    value = row_data[col_letter]
                    if pd.notna(value) and value != "":
                        worksheet_dashboard.write(f'{col_letter}{excel_row+1}', value, subheader_format)
                current_row += 1
            
            # Check for platform rows (DAAS, PAAS, SAAS, OTHER)
            elif row_data['A'] in ["DAAS", "PAAS", "SAAS", "OTHER"]:
                # Write platform row with alternating colors
                row_format = platform_row_format
                if excel_row % 2 == 0:
                    row_format = workbook.add_format({
                        'font_size': 10,
                        'align': 'left',
                        'valign': 'vcenter',
                        'border': 1,
                        'bg_color': '#F2F2F2'  # Light gray for alternating rows
                    })
                
                for col_idx, col_letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']):
                    value = row_data[col_letter]
                    if pd.notna(value) and value != "":
                        worksheet_dashboard.write(f'{col_letter}{excel_row+1}', value, row_format)
                current_row += 1
            
            # Check for total rows (with emoji)
            elif "📊" in str(row_data['A']):
                # Write total row
                for col_idx, col_letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']):
                    value = row_data[col_letter]
                    if pd.notna(value) and value != "":
                        worksheet_dashboard.write(f'{col_letter}{excel_row+1}', value, total_row_format)
                current_row += 1
            
            # Handle empty rows
            else:
                if pd.notna(row_data['A']) and row_data['A'] != "":
                    for col_idx, col_letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']):
                        value = row_data[col_letter]
                        if pd.notna(value):
                            worksheet_dashboard.write(f'{col_letter}{excel_row+1}', value)
                current_row += 1
        
        # Apply conditional formatting for the utilization cells
        # CPU columns (C, D, E)
        cpu_cols = ['C', 'D', 'E']
        for col in cpu_cols:
            # Apply green formatting to Green column
            if col == 'C':
                worksheet_dashboard.conditional_format(f'{col}6:{col}{current_row}', {
                    'type': 'text',
                    'criteria': 'containing',
                    'value': '(',
                    'format': fmt_green
                })
            # Apply amber formatting to Amber column
            elif col == 'D':
                worksheet_dashboard.conditional_format(f'{col}6:{col}{current_row}', {
                    'type': 'text',
                    'criteria': 'containing',
                    'value': '(',
                    'format': fmt_orange
                })
            # Apply red formatting to Red column
            elif col == 'E':
                worksheet_dashboard.conditional_format(f'{col}6:{col}{current_row}', {
                    'type': 'text',
                    'criteria': 'containing',
                    'value': '(',
                    'format': fmt_red
                })
        
        # Memory columns (G, H, I)
        mem_cols = ['G', 'H', 'I']
        for col in mem_cols:
            # Apply green formatting to Green column
            if col == 'G':
                worksheet_dashboard.conditional_format(f'{col}6:{col}{current_row}', {
                    'type': 'text',
                    'criteria': 'containing',
                    'value': '(',
                    'format': fmt_green
                })
            # Apply amber formatting to Amber column
            elif col == 'H':
                worksheet_dashboard.conditional_format(f'{col}6:{col}{current_row}', {
                    'type': 'text',
                    'criteria': 'containing',
                    'value': '(',
                    'format': fmt_orange
                })
            # Apply red formatting to Red column
            elif col == 'I':
                worksheet_dashboard.conditional_format(f'{col}6:{col}{current_row}', {
                    'type': 'text',
                    'criteria': 'containing',
                    'value': '(',
                    'format': fmt_red
                })
        
        # Disk columns (K, L, M) - only if resource has disk metrics
        disk_cols = ['K', 'L', 'M']
        for col in disk_cols:
            # Apply green formatting to Green column
            if col == 'K':
                worksheet_dashboard.conditional_format(f'{col}6:{col}{current_row}', {
                    'type': 'text',
                    'criteria': 'containing',
                    'value': '(',
                    'format': fmt_green
                })
            # Apply amber formatting to Amber column
            elif col == 'L':
                worksheet_dashboard.conditional_format(f'{col}6:{col}{current_row}', {
                    'type': 'text',
                    'criteria': 'containing',
                    'value': '(',
                    'format': fmt_orange
                })
            # Apply red formatting to Red column
            elif col == 'M':
                worksheet_dashboard.conditional_format(f'{col}6:{col}{current_row}', {
                    'type': 'text',
                    'criteria': 'containing',
                    'value': '(',
                    'format': fmt_red
                })
        
        # Auto-adjust column widths for better presentation
        column_widths = {
            'A': 15,  # Platform
            'B': 15,  # Total Deployments
            'C': 15,  # CPU Green
            'D': 15,  # CPU Amber
            'E': 15,  # CPU Red
            'F': 3,   # Spacer
            'G': 15,  # Mem Green
            'H': 15,  # Mem Amber
            'I': 15,  # Mem Red
            'J': 3,   # Spacer
            'K': 15,  # Disk Green
            'L': 15,  # Disk Amber
            'M': 15   # Disk Red
        }
        
        for col_letter, width in column_widths.items():
            worksheet_dashboard.set_column(f'{col_letter}:{col_letter}', width)
        
        # Add freeze panes after headers
        worksheet_dashboard.freeze_panes(6, 0)
        
        # 2. Save RDS/Cache Metrics as separate sheet
        if not metrics_summary.empty:
            metrics_summary.to_excel(writer, index=False, sheet_name="RDS-Cache Metrics")
            worksheet_metrics = writer.sheets["RDS-Cache Metrics"]
            
            # Enhanced formatting for metrics sheet
            metrics_title_format = workbook.add_format({
                'bold': True,
                'font_size': 14,
                'align': 'center',
                'valign': 'vcenter',
                'fg_color': '#8064A2',
                'font_color': 'white'
            })
            
            metrics_header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'align': 'center',
                'fg_color': '#E5E0EC',
                'border': 1,
                'font_size': 11
            })
            
            # Add title
            worksheet_metrics.merge_range('A1:C1', 'RDS & CACHE AGGREGATED METRICS', metrics_title_format)
            
            # Write headers
            for col_num, value in enumerate(metrics_summary.columns.values):
                worksheet_metrics.write(1, col_num, value, metrics_header_format)
            
            # Write data with alternating row colors
            for row_idx in range(len(metrics_summary)):
                data_row = metrics_summary.iloc[row_idx]
                excel_row = row_idx + 2
                
                for col_num, value in enumerate(data_row):
                    if excel_row % 2 == 0:
                        row_color_format = workbook.add_format({'bg_color': '#F2F2F2'})
                    else:
                        row_color_format = workbook.add_format({'bg_color': '#FFFFFF'})
                    worksheet_metrics.write(excel_row, col_num, value, row_color_format)
            
            # Apply conditional formatting for metrics values
            if "Value" in metrics_summary.columns:
                value_col_idx = metrics_summary.columns.get_loc("Value")
                
                # Apply conditional formatting for all rows with numeric values
                for row_idx in range(2, len(metrics_summary) + 2):
                    # Check if this is a metric row (not Total Instances)
                    metric_name = metrics_summary.iloc[row_idx-2]["Metric"]
                    if "CPU/P95" in metric_name or "Mem/P95" in metric_name or "Disk Util/P95" in metric_name:
                        # Apply formatting based on CORRECT value ranges
                        worksheet_metrics.conditional_format(row_idx, value_col_idx, row_idx, value_col_idx,
                                                             {'type': 'cell', 'criteria': 'between', 
                                                              'minimum':0, 'maximum':75, 'format': fmt_green})
                        worksheet_metrics.conditional_format(row_idx, value_col_idx, row_idx, value_col_idx,
                                                             {'type': 'cell', 'criteria': 'between', 
                                                              'minimum':75, 'maximum':89, 'format': fmt_orange})
                        worksheet_metrics.conditional_format(row_idx, value_col_idx, row_idx, value_col_idx,
                                                             {'type': 'cell', 'criteria': 'between', 
                                                              'minimum':90, 'maximum':100, 'format': fmt_red})
            
            # Auto-adjust column widths for metrics
            for i, col in enumerate(metrics_summary.columns):
                try:
                    # Convert column to string, handle NaN values
                    col_str = metrics_summary[col].astype(str).fillna('')
                    column_len = max(col_str.map(len).max(), len(col)) + 4
                    worksheet_metrics.set_column(i, i, column_len)
                except:
                    worksheet_metrics.set_column(i, i, 15)
            
            # Freeze panes
            worksheet_metrics.freeze_panes(2, 0)
        
        # 3. Save individual resource tabs
        for resource_type, df in all_data.items():
            if df.empty:
                continue
                
            if resource_type == "ECS":
                # ECS columns in exact order as requested
                final_columns = [
                    "Application Baseline", "Critical",
                    "Zone A Instance #", "Zone B Instance #",
                    "Application CPU/P95", "Application Mem/P95", "Application Disk Util/P95",
                    "environment", "environment_group", "tribe", "squad", "platform", "alerting",
                    "datetime", "Platform_Category"
                ]
                
                # Keep only columns that exist
                final_columns = [c for c in final_columns if c in df.columns]
                output_df = df[final_columns].copy()
                
                # Save to Excel
                output_df.to_excel(writer, index=False, sheet_name=resource_type, startrow=1)
                worksheet = writer.sheets[resource_type]
                
                # Add ECS sheet title
                ecs_title_format = workbook.add_format({
                    'bold': True,
                    'font_size': 14,
                    'align': 'center',
                    'valign': 'vcenter',
                    'fg_color': '#4F81BD',
                    'font_color': 'white'
                })
                worksheet.merge_range('A1:O1', 'ECS APPLICATION UTILIZATION SUMMARY', ecs_title_format)
                
                # Write headers starting from row 2
                for col_num, value in enumerate(output_df.columns.values):
                    worksheet.write(1, col_num, value, section_header_format)
                
                # Apply alternating row colors for data rows
                for row_idx in range(len(output_df)):
                    excel_row = row_idx + 2
                    if excel_row % 2 == 0:
                        row_color_format = workbook.add_format({'bg_color': '#F2F2F2'})
                    else:
                        row_color_format = workbook.add_format({'bg_color': '#FFFFFF'})
                    
                    for col_num in range(len(output_df.columns)):
                        value = output_df.iloc[row_idx, col_num]
                        worksheet.write(excel_row, col_num, value, row_color_format)
                
                # Highlight Application metrics
                highlight_cols = [
                    "Application CPU/P95", "Application Mem/P95", "Application Disk Util/P95"
                ]
                
                for col in highlight_cols:
                    if col in output_df.columns:
                        col_idx = final_columns.index(col)
                        # Apply conditional formatting with CORRECT ranges
                        worksheet.conditional_format(2, col_idx, len(output_df) + 1, col_idx,
                                                     {'type': 'cell', 'criteria': 'between', 
                                                      'minimum':0, 'maximum':75, 'format': fmt_green})
                        worksheet.conditional_format(2, col_idx, len(output_df) + 1, col_idx,
                                                     {'type': 'cell', 'criteria': 'between', 
                                                      'minimum':75, 'maximum':89, 'format': fmt_orange})
                        worksheet.conditional_format(2, col_idx, len(output_df) + 1, col_idx,
                                                     {'type': 'cell', 'criteria': 'between', 
                                                      'minimum':90, 'maximum':100, 'format': fmt_red})
                
                # Auto-adjust column widths for ECS
                for i, col in enumerate(output_df.columns):
                    try:
                        # Convert column to string, handle NaN values
                        col_str = output_df[col].astype(str).fillna('')
                        column_len = max(col_str.map(len).max(), len(col)) + 2
                        worksheet.set_column(i, i, column_len)
                    except:
                        worksheet.set_column(i, i, 15)
                
                # Freeze panes
                worksheet.freeze_panes(2, 0)
                    
            elif resource_type == "RDS":
                # RDS columns in exact order as requested
                key_cols = []
                
                # Add columns in specified order
                if "Identifier_Name" in df.columns:
                    key_cols.append("Identifier_Name")
                
                # Check for Instance_Class (renamed from Instance_Type)
                if "Instance_Class" in df.columns:
                    key_cols.append("Instance_Class")
                
                # Add NEW columns: vCPUs, Memory_GB, Engine, Max_supported_connections
                if "vCPUs" in df.columns:
                    key_cols.append("vCPUs")
                if "Memory_GB" in df.columns:
                    key_cols.append("Memory_GB")
                if "Engine" in df.columns:
                    key_cols.append("Engine")
                if "Max_supported_connections" in df.columns:
                    key_cols.append("Max_supported_connections")
                
                # Add metrics columns
                metric_cols = ["RDS_Max_CPU_Util", "RDS_P95_CPU", "RDS_Max_Mem_Util", 
                              "RDS_P95_Mem", "RDS_DISK_Usage"]
                for col in metric_cols:
                    if col in df.columns:
                        key_cols.append(col)
                
                # Add remaining columns in specified order
                if "datetime" in df.columns:
                    key_cols.append("datetime")
                
                if "Platform_Category" in df.columns:
                    key_cols.append("Platform_Category")
                
                if "platform" in df.columns:
                    key_cols.append("platform")
                
                output_df = df[key_cols].copy()
                output_df.to_excel(writer, index=False, sheet_name=resource_type, startrow=1)
                
                # Auto-adjust column widths
                worksheet = writer.sheets[resource_type]
                
                # Add RDS sheet title
                rds_title_format = workbook.add_format({
                    'bold': True,
                    'font_size': 14,
                    'align': 'center',
                    'valign': 'vcenter',
                    'fg_color': '#C0504D',
                    'font_color': 'white'
                })
                last_col_letter = chr(65 + len(key_cols) - 1) if len(key_cols) <= 26 else 'Z'
                worksheet.merge_range(f'A1:{last_col_letter}1', 'RDS UTILIZATION DETAILS', rds_title_format)
                
                # Write headers starting from row 2
                for col_num, value in enumerate(output_df.columns.values):
                    worksheet.write(1, col_num, value, section_header_format)
                
                # Apply alternating row colors for data rows
                for row_idx in range(len(output_df)):
                    excel_row = row_idx + 2
                    if excel_row % 2 == 0:
                        row_color_format = workbook.add_format({'bg_color': '#F2F2F2'})
                    else:
                        row_color_format = workbook.add_format({'bg_color': '#FFFFFF'})
                    
                    for col_num in range(len(output_df.columns)):
                        value = output_df.iloc[row_idx, col_num]
                        worksheet.write(excel_row, col_num, value, row_color_format)
                
                for i, col in enumerate(output_df.columns):
                    try:
                        # Convert column to string, handle NaN values
                        col_str = output_df[col].astype(str).fillna('')
                        column_len = max(col_str.map(len).max(), len(col)) + 2
                        worksheet.set_column(i, i, column_len)
                    except:
                        worksheet.set_column(i, i, 15)
                
                # Apply conditional formatting for RDS metrics
                highlight_cols = ["RDS_P95_CPU", "RDS_P95_Mem", "RDS_DISK_Usage"]
                
                for col in highlight_cols:
                    if col in output_df.columns:
                        col_idx = key_cols.index(col)
                        # Apply conditional formatting with CORRECT ranges
                        worksheet.conditional_format(2, col_idx, len(output_df) + 1, col_idx,
                                                     {'type': 'cell', 'criteria': 'between', 
                                                      'minimum':0, 'maximum':75, 'format': fmt_green})
                        worksheet.conditional_format(2, col_idx, len(output_df) + 1, col_idx,
                                                     {'type': 'cell', 'criteria': 'between', 
                                                      'minimum':75, 'maximum':89, 'format': fmt_orange})
                        worksheet.conditional_format(2, col_idx, len(output_df) + 1, col_idx,
                                                     {'type': 'cell', 'criteria': 'between', 
                                                      'minimum':90, 'maximum':100, 'format': fmt_red})
                
                # Freeze panes
                worksheet.freeze_panes(2, 0)
                
            elif resource_type == "Cache":
                # Cache columns in exact order as requested
                key_cols = []
                
                # Add columns in specified order
                if "Identifier_Name" in df.columns:
                    key_cols.append("Identifier_Name")
                
                if "Instance_Type" in df.columns:
                    key_cols.append("Instance_Type")
                
                # Add NEW columns after Instance_Type:
                # Memory_GiB, Instance_Specs_Storage_GiB, Max_Supported_Connections, Max_Connections, P95_Connections
                if "Memory_GiB" in df.columns:
                    key_cols.append("Memory_GiB")
                if "Instance_Specs_Storage_GiB" in df.columns:
                    key_cols.append("Instance_Specs_Storage_GiB")
                if "Max_Supported_Connections" in df.columns:
                    key_cols.append("Max_Supported_Connections")
                if "Max_Connections" in df.columns:
                    key_cols.append("Max_Connections")
                if "P95_Connections" in df.columns:
                    key_cols.append("P95_Connections")
                
                # Add metrics columns
                metric_cols = ["Cache_Max_CPU_Util", "Cache_P95_CPU_Util", 
                              "Cache_Max_Mem_Util", "Cache_P95_Mem_Util"]
                for col in metric_cols:
                    if col in df.columns:
                        key_cols.append(col)
                
                # Add Environment column
                if "Environment" in df.columns:
                    key_cols.append("Environment")
                elif "environment" in df.columns:
                    key_cols.append("environment")
                
                # Add remaining columns in specified order
                if "platform" in df.columns:
                    key_cols.append("platform")
                
                if "datetime" in df.columns:
                    key_cols.append("datetime")
                
                if "Platform_Category" in df.columns:
                    key_cols.append("Platform_Category")
                
                output_df = df[key_cols].copy()
                output_df.to_excel(writer, index=False, sheet_name=resource_type, startrow=1)
                
                # Auto-adjust column widths
                worksheet = writer.sheets[resource_type]
                
                # Add Cache sheet title
                cache_title_format = workbook.add_format({
                    'bold': True,
                    'font_size': 14,
                    'align': 'center',
                    'valign': 'vcenter',
                    'fg_color': '#9BBB59',
                    'font_color': 'white'
                })
                last_col_letter = chr(65 + len(key_cols) - 1) if len(key_cols) <= 26 else 'Z'
                worksheet.merge_range(f'A1:{last_col_letter}1', 'CACHE UTILIZATION DETAILS', cache_title_format)
                
                # Write headers starting from row 2
                for col_num, value in enumerate(output_df.columns.values):
                    worksheet.write(1, col_num, value, section_header_format)
                
                # Apply alternating row colors for data rows
                for row_idx in range(len(output_df)):
                    excel_row = row_idx + 2
                    if excel_row % 2 == 0:
                        row_color_format = workbook.add_format({'bg_color': '#F2F2F2'})
                    else:
                        row_color_format = workbook.add_format({'bg_color': '#FFFFFF'})
                    
                    for col_num in range(len(output_df.columns)):
                        value = output_df.iloc[row_idx, col_num]
                        worksheet.write(excel_row, col_num, value, row_color_format)
                
                for i, col in enumerate(output_df.columns):
                    try:
                        # Convert column to string, handle NaN values
                        col_str = output_df[col].astype(str).fillna('')
                        column_len = max(col_str.map(len).max(), len(col)) + 2
                        worksheet.set_column(i, i, column_len)
                    except:
                        worksheet.set_column(i, i, 15)
                
                # Apply conditional formatting for Cache metrics
                highlight_cols = ["Cache_P95_CPU_Util", "Cache_P95_Mem_Util"]
                
                for col in highlight_cols:
                    if col in output_df.columns:
                        col_idx = key_cols.index(col)
                        # Apply conditional formatting with CORRECT ranges
                        worksheet.conditional_format(2, col_idx, len(output_df) + 1, col_idx,
                                                     {'type': 'cell', 'criteria': 'between', 
                                                      'minimum':0, 'maximum':75, 'format': fmt_green})
                        worksheet.conditional_format(2, col_idx, len(output_df) + 1, col_idx,
                                                     {'type': 'cell', 'criteria': 'between', 
                                                      'minimum':75, 'maximum':89, 'format': fmt_orange})
                        worksheet.conditional_format(2, col_idx, len(output_df) + 1, col_idx,
                                                     {'type': 'cell', 'criteria': 'between', 
                                                      'minimum':90, 'maximum':100, 'format': fmt_red})
                
                # Freeze panes
                worksheet.freeze_panes(2, 0)
        
        # The writer will auto-close when exiting the context manager
    
    # Copy to latest file
    import shutil
    shutil.copy2(out_path, latest_path)
    
    print(f"\nSaved consolidated output: {out_path} (backup: {latest_path})")
    print(f"\nExcel file contains:")
    print(f"  1. Executive Dashboard tab - Executive presentation format")
    if not metrics_summary.empty:
        print(f"  2. RDS-Cache Metrics tab - Aggregated metrics for RDS and Cache")
    tab_num = 3
    for resource_type in all_data.keys():
        if not all_data[resource_type].empty:
            print(f"  {tab_num}. {resource_type} tab - Detailed data in specified order")
            tab_num += 1
    
    # Print executive dashboard preview
    print(f"\nExecutive Dashboard Preview:")
    print(executive_dashboard.to_string(index=False))
    
    # Print RDS/Cache metrics
    if not metrics_summary.empty:
        print(f"\nRDS/Cache Metrics:")
        print(metrics_summary.to_string(index=False))

# -------------------------------
# Main
# -------------------------------

def main():
    print("Starting Multi-Resource Utilization Summary...\n")
    print("="*60)
    
    # Get all resource files
    resource_files = get_resource_files()
    
    if not resource_files:
        print("ERROR: No resource files found in 'raw' directory!")
        print("Expected files: ecs_*.csv, rds_*.csv, cache_*.csv")
        return
    
    print(f"Found {len(resource_files)} resource file(s):")
    for resource_type, file_path in resource_files.items():
        print(f"  • {resource_type}: {os.path.basename(file_path)}")
    print()
    
    all_data = {}
    raw_ecs_data = None  # Store raw ECS data for accurate instance counts
    raw_rds_data = None  # Store raw RDS data
    raw_cache_data = None  # Store raw Cache data
    
    # Process each resource type
    for resource_type, file_path in resource_files.items():
        print(f"Processing {resource_type}...")
        print("-"*40)
        
        # Load and process the file
        df = load_resource_file(file_path, resource_type)
        
        # Filter for production environment
        df_prod = filter_production(df)
        
        # Store raw data for breakdown summary
        if resource_type == "ECS":
            raw_ecs_data = df_prod.copy()
            # Create consolidated view for ECS tab
            df_consolidated = consolidate_ecs(df_prod)
            all_data[resource_type] = df_consolidated
            print(f"ECS: {len(df_prod)} instances consolidated into {len(df_consolidated)} application groups")
            
            # Print instance counts by platform
            if "Platform_Category" in df_prod.columns:
                print("\nECS Instance Counts by Platform (RAW):")
                platform_counts = df_prod["Platform_Category"].value_counts()
                for platform, count in platform_counts.items():
                    print(f"  • {platform}: {count:,} instances")
                print(f"  • TOTAL: {len(df_prod):,} instances")
            
            # Print consolidated application counts by platform
            if "Platform_Category" in df_consolidated.columns:
                print("\nECS Application Counts by Platform (CONSOLIDATED):")
                platform_counts = df_consolidated["Platform_Category"].value_counts()
                for platform, count in platform_counts.items():
                    print(f"  • {platform}: {count:,} applications")
                print(f"  • TOTAL: {len(df_consolidated):,} applications")
        elif resource_type == "RDS":
            raw_rds_data = df_prod.copy()
            all_data[resource_type] = df_prod
            print(f"RDS: {len(df_prod)} instances in production")
            
            # Print RDS metrics preview
            if len(df_prod) > 0:
                metrics = get_resource_metrics(df_prod, "RDS")
                print(f"\nRDS Aggregated Metrics:")
                print(f"  • Total Instances: {metrics['Total Instances']}")
                print(f"  • CPU/P95: {metrics['CPU/P95']}%")
                print(f"  • Mem/P95: {metrics['Mem/P95']}%")
                print(f"  • Disk Util/P95: {metrics['Disk Util/P95']}%")
        elif resource_type == "Cache":
            raw_cache_data = df_prod.copy()
            all_data[resource_type] = df_prod
            print(f"Cache: {len(df_prod)} instances in production")
            
            # Print Cache metrics preview
            if len(df_prod) > 0:
                metrics = get_resource_metrics(df_prod, "Cache")
                print(f"\nCache Aggregated Metrics:")
                print(f"  • Total Instances: {metrics['Total Instances']}")
                print(f"  • CPU/P95: {metrics['CPU/P95']}%")
                print(f"  • Mem/P95: {metrics['Mem/P95']}%")
        
        print()
    
    # Create executive dashboard using the appropriate data
    print("Creating executive dashboard...")
    print("Note: ECS counts are based on CONSOLIDATED applications (post-transformation)")
    print("      RDS/Cache counts are based on individual instances")
    executive_dashboard, metrics_summary = create_executive_dashboard(
        all_data, raw_ecs_data, raw_rds_data, raw_cache_data
    )
    
    # Save all data to Excel
    save_output(all_data, executive_dashboard, metrics_summary)
    
    print("\n" + "="*60)
    print("Done — Multi-resource summary with executive dashboard complete!")
    print("="*60)

if __name__ == "__main__":
    main()