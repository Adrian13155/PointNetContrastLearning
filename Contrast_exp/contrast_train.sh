#!/bin/bash
#SBATCH -J contrast_train       # 作业名
#SBATCH -p gpu                  # 分区
#SBATCH -N 1                    # 节点数
#SBATCH -n 1                    # 任务数
#SBATCH -c 8                    # CPU 核心数
#SBATCH --gres=gpu:1            # 申请1块GPU
#SBATCH -o slurm-%j.out         # 标准输出日志
#SBATCH -e slurm-%j.err         # 错误输出日志

source /data/home/scvi197/run/miniconda3/bin/activate /data/home/scvi197/run/miniconda3/envs/pytorch
python /data/home/scvi197/run/cjj/pointnet.pytorch/Contrast_exp/contrast_train.py
