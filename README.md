# ECS Utilization Summary Script

A Python script that processes ECS utilization data, filters production environments, consolidates metrics by application, and generates an Excel report with zone-based capacity analysis.

## Features

- **Automatic File Detection**: Finds the latest CSV file in the `raw` directory
- **Encoding Detection**: Automatically detects file encoding and delimiter
- **Production Environment Filtering**: Smart filtering of production instances
- **Zone-Based Analysis**: Splits metrics by Availability Zone (A/B)
- **Critical Application Identification**: Flags critical applications
- **Consolidated Metrics**: P95, max utilization, and capacity calculations
- **Excel Output**: Formatted Excel file with conditional highlighting

## Directory Structure

```
project/
├── raw/                          # Input directory
│   ├── ecs_data_20241119.csv     # Your ECS CSV files
│   └── ...
├── transformed/                  # Output directory (created automatically)
│   ├── ecs_utilization_summary_20241119143025.xlsx
│   └── ecs_utilization_summary_latest.xlsx
└── ecs_utilization_script.py     # This script
```

## Prerequisites

### Python Version
- Python 3.6 or higher

### Required Python Packages

Install the required packages using pip:

```bash
pip install pandas chardet numpy xlsxwriter openpyxl
```

#### Package Details:
- **pandas**: Data manipulation and analysis
- **chardet**: Character encoding detection
- **numpy**: Numerical computations
- **xlsxwriter**: Excel file creation with formatting
- **openpyxl**: Excel file reading/writing

## Installation

1. **Install Python 3.6+** if not already installed
2. **Install dependencies**:
   ```bash
   pip install pandas chardet numpy xlsxwriter openpyxl
   ```
3. **Create the directory structure**:
   ```bash
   mkdir raw
   mkdir transformed
   ```
4. **Place your ECS CSV files** in the `raw` directory

## Input File Requirements

### Supported File Format
- **Format**: CSV files
- **Location**: `raw/` directory
- **Naming**: Any name with `.csv` extension

### Required Columns:
The script looks for these columns (case-insensitive):
- `Identifier_Name` or any column containing "id"/"identifier"
- `Tags` (for component extraction)
- `Zone_Id` (for zone classification)
- Utilization metrics: `Max_CPU_Util`, `P95_CPU`, `Max_Mem_Util`, `P95_Mem`, etc.

### Optional Columns:
- `Environment`, `env`, `stage` (for environment filtering)
- `vCPUs`, `Memory_GB`, `diskusage_util` (for capacity calculations)

## Usage

### Step 1: Prepare Input Files
1. Place your ECS CSV files in the `raw/` directory
2. Ensure files contain the required columns

### Step 2: Run the Script
```bash
python ecs_utilization_script.py
```

### Step 3: Check Output
- Output will be saved in `transformed/` directory
- Two files are created:
  - **Timestamped file**: `ecs_utilization_summary_YYYYMMDDHHMMSS.xlsx`
  - **Latest file**: `ecs_utilization_summary_latest.xlsx`

## Expected Output

### Console Output Example
```
Starting ECS Utilization Summary with Zone A/B Split...

Loading ecs_data_20241119.csv ...
   • Detected encoding: UTF-8
   • Using delimiter: ','
   • Loaded 10,000 rows × 25 columns
   • Mapped 'Identifier_Name' → 'identifier'
Filtered production environment rows: 8,500 / 10,000
   • Explicit production rows: 6,000
   • Blank environment rows: 2,500
   • Non-production rows excluded: 1,500
   • Final production rows: 8,500
Saved consolidated output: transformed/ecs_utilization_summary_20241119143025.xlsx (backup: transformed/ecs_utilization_summary_latest.xlsx)

 Done — AZ capacity summary with consolidated metrics complete!
```

### Excel Output Features

#### Columns Included:
- **Application Baseline**: Grouped application name
- **Critical**: Flags critical applications
- **Environment**: Detected environment
- **Capacity Metrics**: Total instances, vCPUs, memory
- **Utilization Metrics**: Max and P95 for CPU, memory, disk
- **Zone Analysis**: Instance counts and metrics per zone (A/B)
- **Metadata**: Tribe, platform, validation columns

#### Conditional Formatting:
- **🟠 Orange**: Values between 75-89%
- **🔴 Red**: Values between 90-100%

## Configuration

### Critical Applications
The script automatically flags these critical applications:
```python
CRITICAL_APPLICATIONS = {
    "apacquireprod", "apbizprod", "apcashier", "apfundprod", "apmobileappng",
    "apmobileprod", "gcfund", "gcrouter", "gcuser", "ifcassetflux",
    "ifccardcenter", "ifcdart", "ifcdatabus", "ifcfluxbatch", "ifcgotone",
    "ifcinnertrans", "ifclimitcenter", "ifcriskcloud", "ifcsupergw",
    "mpaas_mdap", "mpaasgw"
}
```

### Production Filtering Logic
The script includes production instances based on:
- **Explicit production**: Environment contains "prod" or "production"
- **Blank environments**: Missing or empty environment values
- **Excludes**: Development, staging, test, UAT, and other non-prod environments

### Zone Classification
- **Zone A**: Zones ending with 'a' or containing 'sgp1a'/'ap-southeast-1a'
- **Zone B**: Zones ending with 'b' or containing 'sgp1b'/'ap-southeast-1b'
- **Unknown**: All other zones

## Troubleshooting

### Common Issues

1. **"No CSV files found in 'raw' directory"**
   - Ensure CSV files are in the `raw/` directory
   - Check file extensions are `.csv`

2. **Encoding errors**
   - The script automatically detects encoding, but manual override may be needed
   - Check file encoding with a text editor

3. **Missing required columns**
   - Ensure your CSV has `Identifier_Name` or similar identifier column
   - Check for `Tags` column for component extraction

4. **Memory issues with large files**
   - The script processes files in chunks if needed
   - Consider splitting very large files (>1GB)

### Performance Tips
- Place only the latest file in `raw/` directory
- Ensure adequate free memory for large datasets
- Close Excel before running the script to avoid file locks

## Output Analysis

The generated Excel file provides:

1. **Application-Level View**: Consolidated metrics per application
2. **Zone Distribution**: How instances are distributed across AZs
3. **Performance Insights**: P95 and max utilization metrics
4. **Capacity Planning**: Total vCPUs and memory per application
5. **Criticality Focus**: Highlighted critical applications

