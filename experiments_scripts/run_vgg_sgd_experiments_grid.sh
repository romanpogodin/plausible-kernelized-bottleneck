#!/bin/bash
export logs_folder='./logs_vgg_sgd'
mkdir -p "$logs_folder"

trap "exit" INT  # so the loop can be interrupted

for i in 1 2 3 4 5
do

export x=1
export lro=0.0001
export lrb=0.4
export dim=64
export mode='divnorm'
export rule='gauss'
export name="${rule}_${mode}_${dim}_${x}x_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as x=$x, lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_vgg_sgd_experiments_single.sh

export x=1
export lro=0.0005
export lrb=0.5
export dim=64
export mode='divnorm'
export rule='cossim'
export name="${rule}_${mode}_${dim}_${x}x_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as x=$x, lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_vgg_sgd_experiments_single.sh

export x=1
export lro=0.00005
export lrb=0.03
export dim=32
export mode='grouping'
export rule='cossim'
export name="${rule}_${mode}_${dim}_${x}x_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as x=$x, lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_vgg_sgd_experiments_single.sh

export x=1
export lro=0.005
export lrb=0.0
export dim=64
export mode='none'
export rule='backprop'
export name="${rule}_${mode}_${dim}_${x}x_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as x=$x, lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_vgg_sgd_experiments_single.sh

export x=1
export lro=0.006
export lrb=0.0
export dim=64
export mode='divnorm'
export rule='backprop'
export name="${rule}_${mode}_${dim}_${x}x_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as x=$x, lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_vgg_sgd_experiments_single.sh

export x=2
export lro=0.0001
export lrb=0.4
export dim=64
export mode='divnorm'
export rule='gauss'
export name="${rule}_${mode}_${dim}_${x}x_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as x=$x, lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_vgg_sgd_experiments_single.sh

export x=2
export lro=0.0005
export lrb=0.5
export dim=64
export mode='divnorm'
export rule='cossim'
export name="${rule}_${mode}_${dim}_${x}x_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as x=$x, lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_vgg_sgd_experiments_single.sh

export x=2
export lro=0.00005
export lrb=0.03
export dim=32
export mode='grouping'
export rule='cossim'
export name="${rule}_${mode}_${dim}_${x}x_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as x=$x, lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_vgg_sgd_experiments_single.sh

export x=2
export lro=0.006
export lrb=0.0
export dim=64
export mode='none'
export rule='backprop'
export name="${rule}_${mode}_${dim}_${x}x_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as x=$x, lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_vgg_sgd_experiments_single.sh

export x=2
export lro=0.006
export lrb=0.0
export dim=64
export mode='divnorm'
export rule='backprop'
export name="${rule}_${mode}_${dim}_${x}x_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as x=$x, lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_vgg_sgd_experiments_single.sh

done

echo "Done"