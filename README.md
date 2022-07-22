<div align=center>
<img src="./docs/parafoldlogo.png" width="400" >
</div>

# ParallelFold

Author: Bozitao Zhong - zbztzhz@gmail.com

:station: We are adding new functions to ParallelFold, you can see our [Roadmap](https://trello.com/b/sAqBIxBC/parallelfold).

:bookmark_tabs: Please cite our [paper](https://arxiv.org/abs/2111.06340) if you used ParallelFold (ParaFold) in you research. 

## Overview

This project is a modified version of DeepMind's [AlphaFold2](https://github.com/deepmind/alphafold) to achieve high-throughput protein structure prediction. 

We have these following modifications to the original AlphaFold pipeline:

- Divide **CPU part** (MSA and template searching) and **GPU part** (prediction model)

**ParallelFold now supports AlphaFold 2.1.2**



## How to install 

We recommend to install AlphaFold locally, and not using **docker**.

For CUDA 11, you can refer to the [installation guide here](./docs/install.md).

For CUDA 10.1, you can refer to the [installation guide here](./docs/install_cuda10.md).

For ROCM platform as follows,propose use miniconda install parafold environment.  
```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh
mkdir -p ~/miniconda3
sh  Miniconda3-py37_4.9.2-Linux-x86_64.sh
~/miniconda3/bin/conda init
source ~/.bashrc
conda create -n parafold python=3.8
conda install -y -c conda-forge openmm=7.5.1 pdbfixer
conda install -y -c bioconda hmmer hhsuite==3.3.0 kalign2
pip install -r setup/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install /public/software/apps/DeepLearning/whl/dtk-22.04/tensorflow-2.7.0_dtk22.04-cp38-
cp38-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple/

#unload jaxlib and install rocm jaxlib" 
pip uninstall jaxlib -y
pip install /public/software/apps/DeepLearning/whl/dtk-22.04.2/jaxlib-0.3.14-cp38-none-manylinux2014_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple/

#patch for jaxlib
work_path="$PWD"
a="$(which python)"
cd "$(dirname "$(dirname "$a")")/lib/python3.8/site-packages"
patch -p0 < "$work_path/setup/openmm.patch"
patch -p1 < "$work_path/setup/jax_0.3.14_ROCM.patch"
```

## Some detail information of modified files

- `run_alphafold.py`: modified version of original `run_alphafold.py`, it has multiple additional functions like skipping featuring steps when exists `feature.pkl` in output folder
- `run_alphafold.sh`: bash script to run `run_alphafold.py`
- `run_figure.py`: this file can help you make figure for your system



## How to run

Visit the [usage page](./docs/usage.md) to know how to run.  
there is a difference in rocm,"run_alphafold.sh" script can not execute independent. for example     
```bash
python `sh ./run_alphafold.sh \
-d data \
-o output \
-p monomer_ptm \
-i input/GA98.fasta \
-t 1800-01-01 \
-m model_1 \
-f`
```
```bash
python `sh ./run_alphafold.sh \
-d data \
-o output \
-m model_1,model_2,model_3,model_4,model_5 \
-p monomer_ptm \
-i input/GA98.fasta \
-t 1800-01-01`
```
```bash
python `sh ./run_alphafold.sh \
-d data \
-o output \
-m model_1_multimer,model_2_multimer,model_3_multimer,model_4_multimer,model_5_multimer \
-p multimer \
-i input/GA98.fasta \
-t 1800-01-01`
```

## Functions

You can using some flags to change prediction model for ParallelFold:

`-r`: Skip AMBER refinement [Under repair]

`-b`: Using benchmark mode - running JAX model for twice, and the second run can used for evaluate running time

`-R`: Change the number of cycles in recycling

**More functions are under development.**



## What is this for

ParallelFold can help you accelerate AlphaFold when you want to predict multiple sequences. After dividing the CPU part and GPU part, users can finish feature step by multiple processors. Using ParallelFold, you can run AlphaFold 2~3 times faster than DeepMind's procedure. 

**If you have any question, please send GitHub issues**







