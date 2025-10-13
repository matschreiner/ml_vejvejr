#!/usr/bin/env bash
# Copy tarballs, unpack and create HDF5 files (daily)

#source .venv/bin/activate
source /data/projects/glatmodel/uv_envs/ml_env/bin/activate
DPATH=/net/isilon/ifs/arch/home/glatopr/GLATMODEL
PROF_DAY=/data/projects/glatmodel/obs/fild8/road_profiles_h5_daily   # NEW
REPO=/data/projects/glatmodel/repos/ml_vejvejr

YYYY=2022
for MM in $(seq -w 7 12); do
    TBALL=$DPATH/${YYYY}${MM}/fild8/fild8/fild8_${YYYY}${MM}.tar

    process_hourly_file () {
        F="$1"                      # full path to *.gz
        date_part="${F#fild8_}"
        date_part="${date_part%.gz}"
	rawfile=$date_part
        gunzip -c "$F" > $rawfile
        $REPO/scripts/process_profiles_fild8_h5.py $rawfile "$PROF_DAY"
        rm $rawfile
    }

    if [[ -f $TBALL ]]; then
        #tmpdir=$(mktemp -d)
        (   #cd "$tmpdir"
            tar xf "$TBALL"
            cd fild8
            for GZ in *.gz; do
                process_hourly_file "$GZ"
            done
        )
        #rm -r "$tmpdir"
        # optional: monthly merge
        #./merge_daily_to_month_h5.py ${YYYY}${MM} "$PROF_DAY" /data/projects/glatmodel/obs/fild8/h5_monthly
    else
        echo "$TBALL not available"
    fi
done
