#TRAIN=data/road_temp_training_2023.npz
#VALID=data/road_temp_validation_202401.npz
TRAIN=data/road_temp_training_2021-2023.npz
VALID=data/road_temp_val_2024.npz
EPOCH=200
python scripts/train.py $TRAIN $VALID $EPOCH
