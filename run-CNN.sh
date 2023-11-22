# Edit these lines to specify other compute resources, including time. partition, cores, and GPU.
# Time allocation should be in the form d-hh:mm:ss
#!/bin/bash
#SBATCH -p general -q public -t 7-0 -c 12 --mem=128G 
#SBATCH -G a100:1 
#SBATCH -C a100_80
#SBATCH --export=NONE
#SBATCH -o slurm.%j.out

# Not specifying a CNN-variant (e.g. a or c) will result in an error
variant="${1:?-ERROR MUST PASS A LETTER CODE ASSOCIATED WITH A VARIANT}"

# Create a log directory if one does not already exist, and create a txt output file for it.
! [[ -d log ]] && mkdir -p log || :
module load mamba/latest
source activate tensorflow-gpu-2.10.0
python CNN-${variant}.py > log/CNN-${variant}.jid${SLURM_JOB_ID}.txt
