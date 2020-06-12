#!/bin/bash
export logs_folder='./logs_vgg_adam'
mkdir -p "$logs_folder"

trap "exit" INT  # so the loop can be interrupted

for i in 1 2 3 4 5
do

export x=1
export rule='gauss'
export mode='grouping'
export lro=0.000500
export lrb=0.005000
export dim=64
export name="${rule}_${mode}_${dim}_${x}x_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as x=$x, lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_vgg_adam_experiments_single.sh

export x=1
export rule='gauss'
export mode='divnorm'
export lro=0.000500
export lrb=0.010000
export dim=64
export name="${rule}_${mode}_${dim}_${x}x_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as x=$x, lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_vgg_adam_experiments_single.sh

export x=1
export rule='cossim'
export mode='grouping'
export lro=0.000500
export lrb=0.000500
export dim=32
export name="${rule}_${mode}_${dim}_${x}x_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as x=$x, lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_vgg_adam_experiments_single.sh

export x=1
export rule='cossim'
export mode='divnorm'
export lro=0.000500
export lrb=0.000500
export dim=64
export name="${rule}_${mode}_${dim}_${x}x_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as x=$x, lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_vgg_adam_experiments_single.sh

export x=1
export rule='backprop'
mode=''
export lro=0.005000
export lrb=0.500000
export dim=64
export name="${rule}_${mode}_${dim}_${x}x_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as x=$x, lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_vgg_adam_experiments_single.sh

export x=1
export rule='backprop'
export mode='divnorm'
export lro=0.005000
export lrb=0.500000
export dim=64
export name="${rule}_${mode}_${dim}_${x}x_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as x=$x, lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_vgg_adam_experiments_single.sh

export x=2
export rule='gauss'
export mode='grouping'
export lro=0.000500
export lrb=0.005000
export dim=128
export name="${rule}_${mode}_${dim}_${x}x_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as x=$x, lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_vgg_adam_experiments_single.sh

export x=2
export rule='gauss'
export mode='divnorm'
export lro=0.000500
export lrb=0.005000
export dim=128
export name="${rule}_${mode}_${dim}_${x}x_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as x=$x, lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_vgg_adam_experiments_single.sh

export x=2
export rule='cossim'
export mode='grouping'
export lro=0.000500
export lrb=0.000500
export dim=128
export name="${rule}_${mode}_${dim}_${x}x_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as x=$x, lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_vgg_adam_experiments_single.sh

export x=2
export rule='cossim'
export mode='divnorm'
export lro=0.000500
export lrb=0.000500
export dim=64
export name="${rule}_${mode}_${dim}_${x}x_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as x=$x, lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_vgg_adam_experiments_single.sh

export x=2
export rule='backprop'
mode=''
export lro=0.005000
export lrb=0.500000
export dim=64
export name="${rule}_${mode}_${dim}_${x}x_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as x=$x, lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_vgg_adam_experiments_single.sh

export x=2
export rule='backprop'
export mode='divnorm'
export lro=0.005000
export lrb=0.500000
export dim=64
export name="${rule}_${mode}_${dim}_${x}x_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as x=$x, lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_vgg_adam_experiments_single.sh


done
