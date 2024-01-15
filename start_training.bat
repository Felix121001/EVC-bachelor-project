@echo off

echo Starting preprocess_training with config.yaml...
python preprocess_training.py --config_file config.yaml

echo Starting training with config.yaml...
python train.py --config_file config.yaml

echo Training completed.
pause