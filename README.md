# LatentTarget
![cover](assets/overview.png)

## Environment

One can easily install all required packages via 

```
pip install requirements.txt
```

However, a setup script is also provided at `setup_env.sh`, which requires a conda installation and base activation of your environment.

## Data Setup

Provided that your environment setup has been finished, another script called `prepare_data.sh` take you from downloading the data to a training-ready configuration

## Train the Models

```
python train_model.py
```

### Ligand and Protein VAEs

### Latent Diffusion

Note that by default, the latent diffusion model is trained conditioned on protein binding sites within CrossDocked2020, as  

### Pretrained models

Pretrained models can be found within the supplementary files section of the submission.

## Evaluate the Models