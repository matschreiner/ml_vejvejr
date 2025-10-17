#!/usr/bin/env bash
# Copy tarballs, unpack and create HDF5 files (daily)
# Python-parallelized version: Uses multiprocessing in Python for better performance

# ============================================================================
# Configuration and Help
# ============================================================================

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Process fild8 weather data tarballs to HDF5 format using parallel processing.

OPTIONS:
    -h, --help              Show this help message
    -s, --start YYYYMM      Start year-month (e.g., 202201)
    -e, --end YYYYMM        End year-month (e.g., 202212)
    -y, --year YYYY         Process entire year (e.g., 2022)
    -m, --month YYYYMM      Process single month (e.g., 202207)
    -i, --input-dir PATH    Input directory containing tarballs (default: /net/isilon/ifs/arch/home/glatopr/GLATMODEL)
    -o, --output-dir PATH   Output directory for HDF5 files (default: /data/projects/glatmodel/obs/fild8/road_profiles_h5_daily)
    -r, --repo PATH         Repository path (default: /media/cap/extra_work/road_model/ml_vejvejr)
    -w, --workers N         Number of parallel workers (default: all CPU cores)
    -c, --config PATH       Load configuration from file
    --venv PATH             Python virtual environment path (default: /data/projects/glatmodel/uv_envs/ml_env)

EXAMPLES:
    # Process a single month
    $(basename "$0") --month 202207

    # Process a year
    $(basename "$0") --year 2022

    # Process a range of months
    $(basename "$0") --start 202201 --end 202212

    # Process with custom paths
    $(basename "$0") --month 202207 \\
        --input-dir /path/to/tarballs \\
        --output-dir /path/to/output

    # Process with 8 workers
    $(basename "$0") --month 202207 --workers 8

    # Use configuration file
    $(basename "$0") --config convert_fild8.conf --month 202207

ENVIRONMENT VARIABLES:
    WORKERS                 Number of parallel workers (overridden by --workers)

EOF
}

# ============================================================================
# Default Configuration
# ============================================================================

# Default paths
DPATH="/net/isilon/ifs/arch/home/glatopr/GLATMODEL"
PROF_DAY="/data/projects/glatmodel/obs/fild8/road_profiles_h5_daily"
REPO="/media/cap/extra_work/road_model/ml_vejvejr"
VENV_PATH="/data/projects/glatmodel/uv_envs/ml_env"

# Default workers
WORKERS=${WORKERS:-$(nproc)}

# Time period variables
START_YYYYMM=""
END_YYYYMM=""

# Config file
CONFIG_FILE=""

# ============================================================================
# Parse Command Line Arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--start)
            START_YYYYMM="$2"
            shift 2
            ;;
        -e|--end)
            END_YYYYMM="$2"
            shift 2
            ;;
        -y|--year)
            START_YYYYMM="${2}01"
            END_YYYYMM="${2}12"
            shift 2
            ;;
        -m|--month)
            START_YYYYMM="$2"
            END_YYYYMM="$2"
            shift 2
            ;;
        -i|--input-dir)
            DPATH="$2"
            shift 2
            ;;
        -o|--output-dir)
            PROF_DAY="$2"
            shift 2
            ;;
        -r|--repo)
            REPO="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --venv)
            VENV_PATH="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Load Configuration File
# ============================================================================

if [[ -n "$CONFIG_FILE" ]]; then
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "ERROR: Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    echo "Loading configuration from: $CONFIG_FILE"
    source "$CONFIG_FILE"
    echo ""
fi

# ============================================================================
# Validation
# ============================================================================

# Check if time period is specified
if [[ -z "$START_YYYYMM" || -z "$END_YYYYMM" ]]; then
    echo "ERROR: Time period not specified"
    echo "Use --month, --year, or --start/--end to specify the time period"
    echo "Use --help for more information"
    exit 1
fi

# Validate YYYYMM format
if ! [[ "$START_YYYYMM" =~ ^[0-9]{6}$ ]] || ! [[ "$END_YYYYMM" =~ ^[0-9]{6}$ ]]; then
    echo "ERROR: Invalid date format. Use YYYYMM (e.g., 202207)"
    exit 1
fi

# Check if input directory exists
if [[ ! -d "$DPATH" ]]; then
    echo "ERROR: Input directory does not exist: $DPATH"
    exit 1
fi

# Check if repository exists
if [[ ! -d "$REPO" ]]; then
    echo "ERROR: Repository directory does not exist: $REPO"
    exit 1
fi

# Check if Python script exists
if [[ ! -f "$REPO/scripts/process_profiles_fild8_h5_parallel.py" ]]; then
    echo "ERROR: Python script not found: $REPO/scripts/process_profiles_fild8_h5_parallel.py"
    exit 1
fi

# Check if virtual environment exists
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "ERROR: Virtual environment not found: $VENV_PATH"
    exit 1
fi

# ============================================================================
# Setup
# ============================================================================

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Ensure output directory exists
mkdir -p "$PROF_DAY"

# ============================================================================
# Display Configuration
# ============================================================================

echo "=========================================="
echo "fild8 Parallel Processing"
echo "=========================================="
echo "Configuration:"
echo "  Time period:    $START_YYYYMM to $END_YYYYMM"
echo "  Input dir:      $DPATH"
echo "  Output dir:     $PROF_DAY"
echo "  Repository:     $REPO"
echo "  Workers:        $WORKERS"
echo "  Virtual env:    $VENV_PATH"
echo "=========================================="
echo ""

# ============================================================================
# Generate Month List
# ============================================================================

# Extract year and month
START_YYYY=${START_YYYYMM:0:4}
START_MM=${START_YYYYMM:4:2}
END_YYYY=${END_YYYYMM:0:4}
END_MM=${END_YYYYMM:4:2}

# Generate list of months to process
MONTHS=()
CURRENT_YYYY=$START_YYYY
CURRENT_MM=$START_MM

while true; do
    CURRENT_YYYYMM="${CURRENT_YYYY}${CURRENT_MM}"
    MONTHS+=("$CURRENT_YYYYMM")
    
    # Check if we've reached the end
    if [[ "$CURRENT_YYYYMM" == "$END_YYYYMM" ]]; then
        break
    fi
    
    # Increment month
    CURRENT_MM=$((10#$CURRENT_MM + 1))
    if [[ $CURRENT_MM -gt 12 ]]; then
        CURRENT_MM=1
        CURRENT_YYYY=$((CURRENT_YYYY + 1))
    fi
    CURRENT_MM=$(printf "%02d" $CURRENT_MM)
    
    # Safety check to prevent infinite loop
    if [[ $CURRENT_YYYY -gt $((END_YYYY + 1)) ]]; then
        echo "ERROR: Invalid date range"
        exit 1
    fi
done

echo "Processing ${#MONTHS[@]} month(s): ${MONTHS[*]}"
echo ""

# ============================================================================
# Process Each Month
# ============================================================================

PROCESSED_COUNT=0
SKIPPED_COUNT=0
FAILED_COUNT=0

for YYYYMM in "${MONTHS[@]}"; do
    YYYY=${YYYYMM:0:4}
    MM=${YYYYMM:4:2}
    TBALL="$DPATH/${YYYYMM}/fild8/fild8/fild8_${YYYYMM}.tar"

    if [[ ! -f "$TBALL" ]]; then
        echo "WARNING: $TBALL not available, skipping..."
        ((SKIPPED_COUNT++))
        echo ""
        continue
    fi
    
    echo "=========================================="
    echo "Processing tarball: $TBALL"
    echo "=========================================="
    
    # Create a temporary directory for extraction
    tmpdir=$(mktemp -d)
    echo "Using temporary directory: $tmpdir"
    
    # Extract all .gz files from tarball
    echo "Extracting files from tarball..."
    tar xf "$TBALL" -C "$tmpdir" 2>/dev/null
    
    # Count extracted files
    file_count=$(find "$tmpdir/fild8" -name "fild8_*.gz" 2>/dev/null | wc -l)
    
    if [[ $file_count -eq 0 ]]; then
        echo "No files extracted from tarball, skipping..."
        rm -rf "$tmpdir"
        ((SKIPPED_COUNT++))
        echo ""
        continue
    fi
    
    echo "Extracted $file_count file(s)"
    echo ""
    
    # Process all files using the parallel Python script
    if "$REPO/scripts/process_profiles_fild8_h5_parallel.py" \
        "$tmpdir/fild8" \
        "$PROF_DAY" \
        --workers "$WORKERS"; then
        ((PROCESSED_COUNT++))
        echo "✓ Successfully processed ${YYYYMM}"
    else
        ((FAILED_COUNT++))
        echo "✗ Failed to process ${YYYYMM}"
    fi
    
    # Clean up temporary directory
    echo ""
    echo "Cleaning up temporary directory..."
    rm -rf "$tmpdir"
    
    echo ""
done

# ============================================================================
# Summary
# ============================================================================

echo "=========================================="
echo "Processing Complete!"
echo "=========================================="
echo "Summary:"
echo "  Total months:   ${#MONTHS[@]}"
echo "  Processed:      $PROCESSED_COUNT"
echo "  Skipped:        $SKIPPED_COUNT"
echo "  Failed:         $FAILED_COUNT"
echo "=========================================="

# Exit with error if any processing failed
if [[ $FAILED_COUNT -gt 0 ]]; then
    exit 1
fi
