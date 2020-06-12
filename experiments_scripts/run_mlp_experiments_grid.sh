#!/bin/bash
export logs_folder='./logs_mlp'
mkdir -p "$logs_folder"

trap "exit" INT  # so the loop can be interrupted

for i in 1 2 3 4 5
do

export dataset='MNIST'
export rule='gauss'
export mode='none'
export lro=0.000500
export lrb=0.600000
export dim=1
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='MNIST'
export rule='gauss'
export mode='grouping'
export lro=0.000500
export lrb=1.000000
export dim=32
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='MNIST'
export rule='gauss'
export mode='divnorm'
export lro=0.001000
export lrb=1.000000
export dim=32
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='MNIST'
export rule='cossim'
export mode='none'
export lro=0.005000
export lrb=0.500000
export dim=1
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='MNIST'
export rule='cossim'
export mode='grouping'
export lro=0.005000
export lrb=0.600000
export dim=16
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='MNIST'
export rule='cossim'
export mode='divnorm'
export lro=0.005000
export lrb=0.400000
export dim=16
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='MNIST'
export rule='backprop'
export mode='none'
export lro=0.050000
export lrb=0.000000
export dim=1
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='MNIST'
export rule='backprop'
export mode='divnorm'
export lro=0.005000
export lrb=0.000000
export dim=16
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='MNIST'
export rule='last'
export mode='none'
export lro=0.050000
export lrb=0.000000
export dim=1
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='MNIST'
export rule='last'
export mode='divnorm'
export lro=0.050000
export lrb=0.000000
export dim=16
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='fMNIST'
export rule='gauss'
export mode='none'
export lro=0.000500
export lrb=0.500000
export dim=1
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='fMNIST'
export rule='gauss'
export mode='grouping'
export lro=0.000500
export lrb=1.000000
export dim=32
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='fMNIST'
export rule='gauss'
export mode='divnorm'
export lro=0.005000
export lrb=1.000000
export dim=32
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='fMNIST'
export rule='cossim'
export mode='none'
export lro=0.005000
export lrb=1.000000
export dim=1
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='fMNIST'
export rule='cossim'
export mode='grouping'
export lro=0.001000
export lrb=0.600000
export dim=32
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='fMNIST'
export rule='cossim'
export mode='divnorm'
export lro=0.000500
export lrb=1.000000
export dim=32
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='fMNIST'
export rule='backprop'
export mode='none'
export lro=0.005000
export lrb=0.000000
export dim=1
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='fMNIST'
export rule='backprop'
export mode='divnorm'
export lro=0.005000
export lrb=0.000000
export dim=32
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='fMNIST'
export rule='last'
export mode='none'
export lro=0.050000
export lrb=0.000000
export dim=1
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='fMNIST'
export rule='last'
export mode='divnorm'
export lro=0.050000
export lrb=0.000000
export dim=32
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='kMNIST'
export rule='gauss'
export mode='none'
export lro=0.001000
export lrb=0.600000
export dim=1
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='kMNIST'
export rule='gauss'
export mode='grouping'
export lro=0.001000
export lrb=1.000000
export dim=32
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='kMNIST'
export rule='gauss'
export mode='divnorm'
export lro=0.001000
export lrb=1.000000
export dim=32
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='kMNIST'
export rule='cossim'
export mode='none'
export lro=0.005000
export lrb=0.600000
export dim=1
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='kMNIST'
export rule='cossim'
export mode='grouping'
export lro=0.005000
export lrb=0.400000
export dim=16
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='kMNIST'
export rule='cossim'
export mode='divnorm'
export lro=0.000500
export lrb=0.400000
export dim=16
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='kMNIST'
export rule='backprop'
export mode='none'
export lro=0.050000
export lrb=0.000000
export dim=1
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='kMNIST'
export rule='backprop'
export mode='divnorm'
export lro=0.050000
export lrb=0.000000
export dim=32
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='kMNIST'
export rule='last'
export mode='none'
export lro=0.050000
export lrb=0.000000
export dim=1
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='kMNIST'
export rule='last'
export mode='divnorm'
export lro=0.050000
export lrb=0.000000
export dim=16
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='CIFAR10'
export rule='gauss'
export mode='none'
export lro=0.005000
export lrb=0.100000
export dim=1
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='CIFAR10'
export rule='gauss'
export mode='grouping'
export lro=0.000500
export lrb=0.600000
export dim=32
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='CIFAR10'
export rule='gauss'
export mode='divnorm'
export lro=0.001000
export lrb=1.000000
export dim=32
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='CIFAR10'
export rule='cossim'
export mode='none'
export lro=0.001000
export lrb=1.000000
export dim=1
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='CIFAR10'
export rule='cossim'
export mode='grouping'
export lro=0.005000
export lrb=0.400000
export dim=32
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='CIFAR10'
export rule='cossim'
export mode='divnorm'
export lro=0.005000
export lrb=0.100000
export dim=32
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='CIFAR10'
export rule='backprop'
export mode='none'
export lro=0.005000
export lrb=0.000000
export dim=1
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='CIFAR10'
export rule='backprop'
export mode='divnorm'
export lro=0.005000
export lrb=0.000000
export dim=32
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='CIFAR10'
export rule='last'
export mode='none'
export lro=0.050000
export lrb=0.000000
export dim=1
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh

export dataset='CIFAR10'
export rule='last'
export mode='divnorm'
export lro=0.010000
export lrb=0.000000
export dim=32
export name="${dataset}_${rule}_${mode}_${i}.out"
export log_filename="${logs_folder}/${name}"
echo "running as dataset=$dataset,lro=$lro, lrb=$lrb, dim=$dim, mode=$mode, rule=$rule"

./experiments_scripts/run_mlp_experiments_single.sh


done

echo 'Done'
