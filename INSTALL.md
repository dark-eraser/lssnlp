On new GP machines, you need to install the following packages:

- `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
- `bash Miniconda3-latest-Linux-x86_64.sh` (accept everything default)
- `reload shell`
- `conda create --solver=libmamba -n rapids -c rapidsai -c conda-forge -c nvidia rapids=23.08 python=3.10 cuda-version=11.8`
