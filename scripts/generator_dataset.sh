#!/bin/bash

#SBATCH--job-name=rte_data
#SBATCH--partition=cpu
#SBATCH-n10
#SBATCH--cpus-per-task=2
#SBATCH--mem=60G
#SBATCH--output=%j.out
#SBATCH--error=%j.err
#SBATCH--mail-type=all
#SBATCH--mail-user=yourmail-address

# module add matlab/2020a

cd /workspaces/deeprte/generator/2d-sweeping

matlab -nodisplay -r "run train_1.m; exit"
matlab -nodisplay -r "run train_2.m; exit"
