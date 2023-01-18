#!/bin/bash

###
# sbatch-supervised.sh
# This script runs python from within our conda env as a slurm batch job.
# All arguments passed to this script are passed directly to the python
# interpreter.
#

###
# Example usage:
# Running any other python script myscript.py with arguments
# ./sbatch-supervised.sh myscript.py --arg1 --arg2=val2
#

###
# Parameters for sbatch
#
NUM_NODES=1
NUM_CORES=2
NUM_GPUS=2
NUM_TASKS=2
JOB_NAME="Vit"
MAIL_USER="tom.rahav@campus.technion.ac.il"
MAIL_TYPE=END # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=master
PARTITION="gipmed"
MY_ACCOUNT="gipmed"

sbatch \
	-A $MY_ACCOUNT \
    -p $PARTITION \
	-N $NUM_NODES \
	-c $NUM_CORES \
    -n $NUM_TASKS \
	--gres=gpu:$NUM_GPUS \
	--job-name $JOB_NAME \
	--mail-user $MAIL_USER \
	--mail-type $MAIL_TYPE \
	-o 'slurm-gimped-vit-ssl.out' \
<<EOF
#!/bin/bash
# Setup the conda env
conda activate $CONDA_ENV
source $CONDA_HOME/etc/profile.d/conda.sh
echo "*** Activating environment $CONDA_ENV ***"
# export your required environment variables below
export WANDB_API_KEY=a8edcfc5767b7efdd613b561c389b6951eee54e7
# Run python with the args to the script
torchrun --nproc_per_node=2 $@
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"
EOF

