#!/bin/bash

###
# py-sbatch.sh
# This script runs python from within our conda env as a slurm batch job.
# All arguments passed to this script are passed directly to the python
# interpreter.
#

###
# Example usage:
# Running any other python script myscript.py with arguments
# ./py-sbatch.sh myscript.py --arg1 --arg2=val2
#

###
# Parameters for sbatch
#
NUM_NODES=1
NUM_CORES=2
NUM_GPUS=2
NUM_TASKS=2
JOB_NAME="_500_epochs"
MAIL_USER="noam.moshe@campus.technion.ac.il"
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
	-o 'slurm-gimped-vit-supervised.out' \
<<EOF
#!/bin/bash
# Setup the conda env
conda activate $CONDA_ENV
source $CONDA_HOME/etc/profile.d/conda.sh
echo "*** Activating environment $CONDA_ENV ***"
#export your required environment variables below
export WANDB_API_KEY=your_api_key
# Run python with the args to the script
torchrun --nproc_per_node=2 $@
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"
EOF

