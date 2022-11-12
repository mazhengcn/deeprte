#!/bin/bash

#SBATCH--job-name=rte_data
#SBATCH--partition=cpu
#SBATCH-n10
#SBATCH--cpus-per-task=2
#SBATCH--mem=60G
#SBATCH--output=%j.out
#SBATCH--error=%j.err
#SBATCH--mail-type=all
#SBATCH--mail-user=zhuyekun123@sjtu.edu.cn
