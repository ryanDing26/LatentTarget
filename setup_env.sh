#!/bin/bash
# Note: should already have the base conda environment activated.
conda create -n latenttarget -y
conda activate latenttarget
# this ones only needed if you don't have pip by default
conda install pip -y
pip install -r requirements.txt