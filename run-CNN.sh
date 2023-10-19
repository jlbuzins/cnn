#!/bin/bash
#SBATCH -p general -q public -t 7-0 -c 12 --mem=128G 
#SBATCH -G a100:1 
#SBATCH -C a100_80
#SBATCH --export=NONE
#SBATCH -o slurm.%j.out

variant="${1:?-ERROR MUST PASS A LETTER CODE ASSOCIATED WITH A VARIANT}"

! [[ -d log ]] && mkdir -p log || :
module load mamba/latest
source activate tensorflow-gpu-2.10.0
python CNN-${variant}.py > log/CNN-${variant}.jid${SLURM_JOB_ID}.txt
