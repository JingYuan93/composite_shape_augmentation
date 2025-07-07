
#!/usr/bin/env bash

python3 augmentation.py \
  --data_path combined_matrix.csv \
  --num_raw_samples 3 \
  --seq_len 10 \
  --noise_dim 64 \
  --hid_dim 128 \
  --z_dim 64 \
  --TIMESTEPS 1000 \
  --BETA_START 1e-4 \
  --BETA_END 0.02 \
  --LR 1e-5 \
  --EPOCHS 200 \
  --BATCH_SZ 8 \
  --seed 42\
  --lambda_noise 1.0 \
  --lambda_recon 1.0 \
  --lambda_mono 0.1 \
  --lambda_curv 0.1 \
  --lambda_range 1.0 \
  --lambda_interp 10.0 \
  --grad_clip 1.0 \
  --num_samples 30 \
  --out_dir generated \
  --device cuda
