```bash
conda create --name physionet-2025-autogluon python=3.11
conda activate physionet-2025-autogluon
conda install -c conda-forge mamba
mamba install -c conda-forge autogluon "pytorch=*=cuda*"
mamba install -c conda-forge "ray-tune >=2.10.0,<2.32" "ray-default >=2.10.0,<2.32"
```
