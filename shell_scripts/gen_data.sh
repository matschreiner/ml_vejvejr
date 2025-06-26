#!/usr/bin/env bash
#
# validation test
YEARS=2024
STATION="0-100000-0"
DATA_TYPE=val
OUT=/media/cap/extra_work/road_model/ml_vejvejr/data
DATA="/data/projects/glatmodel/obs/fild8/road_profiles_daily"


# training test
YEARS="2021,2022,2023"
DATA_TYPE=train
python scripts/generate_data_from_profiles.py $YEARS $STATION $DATA_TYPE $OUT $DATA
