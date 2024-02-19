#!/bin/bash

gpu=$1
prompt=$2

echo "CUDA:$gpu, Prompt: $prompt"

filename=$(echo "$prompt" | sed 's/ /-/g')
n_particles=1
prompt="A {}."
sketch_path="../pseudosketch_images/n01858441/n01858441_10538.JPEG"
inversion_ckpt="../checkpoints/inversion_16000.pt"

CUDA_VISIBLE_DEVICES=$gpu python main.py --text "$prompt" --sketch_path "$sketch_path" --inversion_ckpt "$inversion_ckpt" --iters 25000 --lambda_entropy 10 --scale 7.5 --n_particles $n_particles --h 512  --w 512 --t5_iters 5000 --workspace exp-nerf-stage1/

# penguin
CUDA_VISIBLE_DEVICES=0 python main.py --text "A {}." --sketch_path "../dataset/pseudosketches/n02056570/n02056570_8709.JPEG" --inversion_ckpt "../checkpoints/inversion_16000.pt" --iters 25000 --lambda_entropy 10 --scale 7.5 --n_particles 1 --h 256  --w 256 --t5_iters 5000 --workspace exp-nerf-stage1/
CUDA_VISIBLE_DEVICES=0 python main.py --text "A photo of{}." --sketch_path "../dataset/pseudosketches/n07739125/n07739125_10672.JPEG" --inversion_ckpt "../checkpoints/inversion_16000.pt" --iters 25000 --lambda_entropy 10 --scale 7.5 --n_particles 1 --h 256  --w 256 --t5_iters 5000 --workspace exp-nerf-stage1/

# pineapple
CUDA_VISIBLE_DEVICES=0 python main.py --text "A {}." --sketch_path "../dataset/pseudosketches/n07753275/n07753275_407.JPEG" --inversion_ckpt "../checkpoints/inversion_16000.pt" --iters 25000 --lambda_entropy 10 --scale 7.5 --n_particles 1 --h 256  --w 256 --t5_iters 5000 --workspace exp-nerf-stage1/

# car
CUDA_VISIBLE_DEVICES=0 python main.py --text "A {}." --sketch_path "../dataset/pseudosketches/n04166281/n02958343_11365.JPEG" --inversion_ckpt "../checkpoints/inversion_16000.pt" --iters 25000 --lambda_entropy 10 --scale 7.5 --n_particles 1 --h 128  --w 128 --t5_iters 5000 --workspace exp-nerf-stage1/

# spider
CUDA_VISIBLE_DEVICES=0 python main.py --text "A {}." --sketch_path "../dataset/pseudosketches/n01772222/n01772222_4754.JPEG" --inversion_ckpt "../checkpoints/inversion_16000.pt" --iters 25000 --lambda_entropy 10 --scale 7.5 --n_particles 1 --h 128  --w 128 --t5_iters 5000 --workspace exp-nerf-stage1/

# tiger
CUDA_VISIBLE_DEVICES=0 python main.py --text "A {}." --sketch_path "../dataset/pseudosketches/n02129604/n02129604_1436.JPEG" --inversion_ckpt "../checkpoints/inversion_16000.pt" --iters 25000 --lambda_entropy 10 --scale 7.5 --n_particles 1 --h 128  --w 128 --t5_iters 5000 --workspace exp-nerf-stage1/

# horse
CUDA_VISIBLE_DEVICES=0 python main.py --text "A cartoon style {}." --sketch_path "../dataset/pseudosketches/n02374451/n02374451_17474.JPEG" --inversion_ckpt "../checkpoints/inversion_16000.pt" --iters 25000 --lambda_entropy 10 --scale 7.5 --n_particles 1 --h 128  --w 128 --t5_iters 5000 --workspace exp-nerf-stage1/

# duck
CUDA_VISIBLE_DEVICES=0 python main.py --text "A {} toy." --sketch_path "../dataset/pseudosketches/n01846331/n01846331_3060.JPEG" --inversion_ckpt "../checkpoints/inversion_16000.pt" --iters 25000 --lambda_entropy 10 --scale 7.5 --n_particles 1 --h 128  --w 128 --t5_iters 5000 --workspace exp-nerf-stage1/
