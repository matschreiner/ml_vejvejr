#!/usr/bin/env bash
# Copy tarballs, unpack and create HDF5 files (daily)
# Python-parallelized version: Uses multiprocessing in Python for better performance

#source .venv/bin/activate
source /data/projects/glatmodel/uv_envs/ml_env/bin/activate
DPATH=/net/isilon/ifs/arch/home/glatopr/GLATMODEL
PROF_DAY=/data/projects/glatmodel/obs/fild8/road_profiles_h5_daily
REPO=/data/projects/glatmodel/repos/ml_vejvejr
REPO=/media/cap/extra_work/road_model/ml_vejvejr

# Number of parallel workers (default: all CPU cores)
WORKERS=${WORKERS:-$(nproc)}

# Ensure output directory exists
mkdir -p "$PROF_DAY"

YYYY=2022
for MM in $(seq -w 08 08); do
    TBALL=$DPATH/${YYYY}${MM}/fild8/fild8/fild8_${YYYY}${MM}.tar

    if [[ ! -f $TBALL ]]; then
        echo "WARNING: $TBALL not available, skipping..."
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
        continue
    fi
    
    echo "Extracted $file_count file(s)"
    echo ""
    
    # Process all files using the parallel Python script
    $REPO/scripts/process_profiles_fild8_h5_parallel.py \
        "$tmpdir/fild8" \
        "$PROF_DAY" \
        --workers "$WORKERS"
    
    # Clean up temporary directory
    echo ""
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
