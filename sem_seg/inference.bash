#!/usr/bin/env bash

python batch_inference.py --model_path=log/model.ckpt           \
  --dump_dir=log/dump --output_filelist log/output_filelist.txt \
  --room_data_filelist shapes/all_data_label.txt --visu
