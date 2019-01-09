#!/usr/bin/env bash

flags="--model_save=model.ckpt --log_dir=log --test_area=test --max_epoch=150"

if [ -f "log/model.ckpt.index" ]; then
  echo "loading log/model.ckpt"
  flags="$flags --model_path=log/model.ckpt"
else
  echo "randomly initializing model weights"
  flags="$flags --new_model=True"
fi

python train.py $flags
