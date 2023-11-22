This repository contains two versions of a convolutional neural net, designed to accept data in the form of 360x3000 pixel images. Given an image of a stratified inclined duct experiment, the CNN will predict the type of turbulence, based on the clusters proposed in Data-driven classification of sheared stratified turbulence, by A. Lefauve and M. Couchman. Version a, CNN-a.py, has 50% dropout, while version b, CNN-b.py, has 60% dropout.
The repository also contains a shell script to run both .py files on the Sol supercomputer. It specifies partition, time limit, and compute resources, and creates a log directory for the output, a text file titled CNN-{variant}.jid${SLURM_JOB_ID}.txt. To use the shell script, enter "{variant} run-CNN.sh" into the Sol terminal.
