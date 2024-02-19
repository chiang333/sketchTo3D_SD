Sketch to 3D generation
===
A practice for stable diffusion application implementation.

### Data prepare
1. (For train inversion module) Download sketch dataset from [Pseudosketches](https://www.dropbox.com/sh/mfogqa8xlzy6mdk/AABDRO_cLMxTVuRm2RAHHOnza?dl=0)
2. Download our trained inversion adapter from [drive](https://drive.google.com/file/d/14P7QpDJvBLN05dHTm3uMz1f-QqQtCiuJ/view?usp=drive_link) put in `checkpoints/`

### Environment
Use the requirements.txt from Prolificdreamer.
``` bash!
conda create -n CG python=3.11
conda activate CG
# install pytorch of your cuda version
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
cd prolidicfreamer
pip install -r requirements.txt
```

### file structure
```
.
├── checkpoints
│   └── inversion_16000.pt
├── dataset
│   ├── pseudosketches
│   └── pseudosketch_images
├── prolificdreamer/
├── README.md
├── sample/
└── textual_inversion_adapter.ipynb
```


## Step 1 - Textual Inversion
Train textual inversion module in `textual_inversion_adapter.ipynb`.
The code is original from [textual_inversion_colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb) and also reference to [train_inversion_adapter.py](https://github.com/miccunifi/ladi-vton/blob/master/src/train_inversion_adapter.py#L19) for adpter design.

## Step 2 - 3D model generation
We modify the code from [ProlificDreamer](https://github.com/thu-ml/prolificdreamer).
### What we modified (in prolific dreamer)
1. `main.py`, nerf/`sd.py` `util.py`
2. add `run_sketch.sh`
### Run
``` bash!
cd prolificdreamer
# bird
CUDA_VISIBLE_DEVICES=0 python main.py --text "A {}." --sketch_path "../sample.JPEG" --inversion_ckpt "../checkpoints/inversion_16000.pt" --iters 25000 --lambda_entropy 10 --scale 7.5 --n_particles 1 --h 256  --w 256 --t5_iters 5000 --workspace exp-nerf-stage1/
```
Use command in `run_sketch.sh` to get other results.
Can try to modify the text prompt and input sketch.
