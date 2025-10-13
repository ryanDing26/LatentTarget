# Credits

- **Data**: Courtesy of [CrossDocked2020 Datasets](https://bits.csb.pitt.edu/files/crossdock2020/) [1] . v1.1 is used instead of the most recent v1.3 in establishing a common benchmark with other models.
- **Preprocessing Scripts**: Courtesy of [TargetDiff](https://github.com/guanjq/targetdiff) [2]. Used for preprocessing in this project.

## Data Preparation

Taken from [2]:

> If you want to process the dataset from scratch, you need to download CrossDocked2020 v1.1 from [here](https://bits.csb.pitt.edu/files/crossdock2020/), save it into `data/CrossDocked2020`, and run the scripts in `scripts/data_preparation`:
> * [clean_crossdocked.py](data_preparation/clean_crossdocked.py) will filter the original dataset and keep the ones with RMSD < 1A.
> It will generate a `index.pkl` file and create a new directory containing the original filtered data (corresponds to `crossdocked_v1.1_rmsd1.0.tar.gz` in the drive). *You don't need these files if you have downloaded .lmdb file.*
    ```bash
    python scripts/data_preparation/clean_crossdocked.py --source data/CrossDocked2020 --dest data/crossdocked_v1.1_rmsd1.0 --rmsd_thr 1.0
    ```
> * [extract_pockets.py](data_preparation/extract_pockets.py) will clip the original protein file to a 10A region around the binding molecule. E.g.
    ```bash
    python scripts/data_preparation/extract_pockets.py --source data/crossdocked_v1.1_rmsd1.0 --dest data/crossdocked_v1.1_rmsd1.0_pocket10
    ```
> * [split_pl_dataset.py](data_preparation/split_pl_dataset.py) will split the training and test set. We use the same split `split_by_name.pt` as 
> [AR](https://arxiv.org/abs/2203.10446) and [Pocket2Mol](https://arxiv.org/abs/2205.07249), which can also be downloaded in the Google Drive - data folder.
    ```bash
    python scripts/data_preparation/split_pl_dataset.py --path data/crossdocked_v1.1_rmsd1.0_pocket10 --dest data/crossdocked_pocket10_pose_split.pt --fixed_split data/split_by_name.pt
    ```

Note: These scripts are meant to be ran from the repository-level directory. Additionally, a comprehensive setup script can be found at `prepare_data.sh`.

## Citations

[1] Francoeur, P. G., Masuda, T., Sunseri, J., Jia, A., Iovanisci, R. B., Snyder, I., and Koes, D. R. Three-dimensional convolutional neural networks and a cross-docked data set for structure-based drug design. Journal of Chemical Information and Modeling, 60(9):4200–4215, 2020.

[2] Guan, J., Qian, W. W., Peng, X., Su, Y., Peng, J., and Ma, J. 3d equivariant diffusion for target-aware molecule generation and affinity prediction. arXiv
preprint arXiv:2303.03543, 2023.