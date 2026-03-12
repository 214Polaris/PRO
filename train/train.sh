#!/bin/sh
#$ -cwd
#$ -l node_f=1                         #specify a type of node              
#$ -l h_rt=24:00:00                   #specify the maximum runtime    maximum time is 24:00:00
#$ -o stdout.log                      #specify the stdout file
#$ -e stderr.log                      #specify the stderr file
#$ -p -3
#$ -m abe                             #specify the conditions for sending email
#$ -M 214wadechen@gmail.com             #mail address

module purge

# load cuda module
module load cuda
# load Intel Compiler
module load intel

# enable conda in non-interactive shells
source /gs/bs/tga-TDSAI/Chen/conda/etc/profile.d/conda.sh

# activate conda env
conda activate pro-h100

# run training
bash train/train_ultrafeedback.sh <exp_id> 8 \
  list_ultrafeedback_train_len8 \
  list_ultrafeedback_dev_len8 \
  list_ultrafeedback_test_len8
