@echo off


python update_yaml.py config_diff.yaml SOLVER.RUN_NAME "Diffusion_5000_timesteps_0.1_lr_1.0e-05" SOLVER.TIMESTEPS 5000 SOLVER.BETA_MAX 0.1 SOLVER.LR 1.0e-05
python train_diff.py config_diff.yaml

python update_yaml.py config_diff.yaml SOLVER.RUN_NAME "Diffusion_5000_timesteps_0.1_lr_1.0e-04" SOLVER.TIMESTEPS 5000 SOLVER.BETA_MAX 0.1 SOLVER.LR 1.0e-04
python train_diff.py config_diff.yaml

python update_yaml.py config_diff.yaml SOLVER.RUN_NAME "Diffusion_5000_timesteps_0.1_lr_5.0e-05" SOLVER.TIMESTEPS 5000 SOLVER.BETA_MAX 0.1 SOLVER.LR 5.0e-05
python train_diff.py config_diff.yaml






