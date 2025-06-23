TRAIN=data/road_temp_training_2023.npz
VALID=data/road_temp_validation_202401.npz
EPOCH=200
python scripts/train.py $TRAIN $VALID $EPOCH
