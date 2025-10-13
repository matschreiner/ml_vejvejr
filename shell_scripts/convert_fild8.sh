#!/usr/bin/env bash
# Copy tarballs, unpack and create HDF5 files (daily)
# Optimized version: skips already-processed files before extraction

#source .venv/bin/activate
source /data/projects/glatmodel/uv_envs/ml_env/bin/activate
DPATH=/net/isilon/ifs/arch/home/glatopr/GLATMODEL
PROF_DAY=/data/projects/glatmodel/obs/fild8/road_profiles_h5_daily   # NEW
REPO=/data/projects/glatmodel/repos/ml_vejvejr

# Ensure output directory exists
mkdir -p "$PROF_DAY"

check_if_processed() {
    local date_part="$1"
    local day_tag="${date_part:0:8}"  # Extract YYYYMMDD
    local h5_file="$PROF_DAY/road_temp_${day_tag}.h5"
    
    # If HDF5 file doesn't exist, definitely not processed
    if [[ ! -f "$h5_file" ]]; then
        return 1  # Not processed
    fi
    
    # Check if timestamp exists in HDF5 file
    python3 -c "
import h5py
import sys
try:
    with h5py.File('$h5_file', 'r') as f:
        if 'dynamic/temp_profile/timestamps_iso' in f:
            timestamps = [ts.decode() for ts in f['dynamic/temp_profile/timestamps_iso'][:]]
            # Check if this hour is already in the file
            for ts in timestamps:
                if '$date_part' in ts:
                    sys.exit(0)  # Already exists
    sys.exit(1)  # Doesn't exist
except Exception as e:
    sys.exit(1)  # Error or doesn't exist
" 2>/dev/null
    return $?
}

process_hourly_file () {
    local F="$1"                      # full path to *.gz
    local date_part="${F#fild8_}"
    date_part="${date_part%.gz}"
    local rawfile="$date_part"
    
    gunzip -c "$F" > "$rawfile"
    $REPO/scripts/process_profiles_fild8_h5.py "$rawfile" "$PROF_DAY"
    local exit_code=$?
    rm -f "$rawfile"
    return $exit_code
}

YYYY=2022
for MM in $(seq -w 7 12); do
    TBALL=$DPATH/${YYYY}${MM}/fild8/fild8/fild8_${YYYY}${MM}.tar

    if [[ ! -f $TBALL ]]; then
        echo "WARNING: $TBALL not available, skipping..."
        continue
    fi
    
    echo "=========================================="
    echo "Processing tarball: $TBALL"
    echo "=========================================="
    
    # List all files in tarball and check which ones need processing
    echo "Analyzing tarball contents..."
    files_to_process=()
    
    while IFS= read -r file; do
        # Extract only .gz files from fild8/ directory
        if [[ "$file" == fild8/fild8_*.gz ]]; then
            basename_file="${file#fild8/}"
            date_part="${basename_file#fild8_}"
            date_part="${date_part%.gz}"
            
            if check_if_processed "$date_part"; then
                echo "  ✓ Already processed: $basename_file"
            else
                echo "  → Need to process: $basename_file"
                files_to_process+=("$file")
            fi
        fi
    done < <(tar tf "$TBALL")
    
    # Process only the files that need processing
    if [[ ${#files_to_process[@]} -eq 0 ]]; then
        echo "All files already processed for ${YYYY}${MM}, skipping extraction."
        continue
    fi
    
    echo ""
    echo "Extracting and processing ${#files_to_process[@]} file(s)..."
    
    # Create a temporary directory for extraction
    tmpdir=$(mktemp -d)
    echo "Using temporary directory: $tmpdir"
    
    # Extract only the files we need
    (
        cd "$tmpdir"
        
        # Extract only the needed files
        for file in "${files_to_process[@]}"; do
            tar xf "$TBALL" "$file" 2>/dev/null
        done
        
        cd fild8
        
        # Process each extracted file
        processed=0
        failed=0
        for GZ in *.gz; do
            [[ ! -f "$GZ" ]] && continue
            
            echo ""
            echo "Processing: $GZ"
            if process_hourly_file "$GZ"; then
                processed=$((processed + 1))
            else
                failed=$((failed + 1))
                echo "  ERROR: Failed to process $GZ"
            fi
        done
        
        echo ""
        echo "Summary: $processed processed, $failed failed"
    )
    
    # Clean up temporary directory
    echo "Cleaning up temporary directory..."
    rm -rf "$tmpdir"
    
    echo "Completed processing for ${YYYY}${MM}"
    echo ""
    
    # optional: monthly merge
    #./merge_daily_to_month_h5.py ${YYYY}${MM} "$PROF_DAY" /data/projects/glatmodel/obs/fild8/h5_monthly
done

echo "=========================================="
echo "All months processed!"
echo "=========================================="
