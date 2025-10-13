#!/bin/bash

################################
# Setup installation directory #
################################

mkdir data
cd data
mkdir CrossDocked2020
cd CrossDocked2020

##########################################################
# Download CrossDockedv1.1 and corresponding types files #
##########################################################

wget https://bits.csb.pitt.edu/files/crossdock2020/v1.1/CrossDocked2020_v1.1.tgz
wget https://bits.csb.pitt.edu/files/crossdock2020/v1.1/CrossDocked2020_v1.1_types.tar.gz

############################
# Extract compressed files #
############################

tar -xvzf CrossDocked2020_v1.1_types.tar.gz
tar -xzvf CrossDocked2020_v1.1.tgz

################################
# Change into proper directory #
################################

cd ../..

##########################
# Preprocess via scripts #
##########################

python scripts/data_preparation/clean_crossdocked.py --source data/CrossDocked2020 --dest data/crossdocked_v1.1_rmsd1.0 --rmsd_thr 1.0
python scripts/data_preparation/extract_pockets.py --source data/crossdocked_v1.1_rmsd1.0 --dest data/crossdocked_v1.1_rmsd1.0_pocket10
python scripts/data_preparation/split_pl_dataset.py --path data/crossdocked_v1.1_rmsd1.0_pocket10 --dest data/crossdocked_pocket10_pose_split.pt --fixed_split data/split_by_name.pt