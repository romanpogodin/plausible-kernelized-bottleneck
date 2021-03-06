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

# new experiments with feedback alignment, sign symmetry and layer-wise classification

echo 'Running cross-entropy_none_1'
log_filename="${logs_folder}/cross-entropy_none_1.out"
python3 experiments.py --batch-size 128 --bottleneck-gamma 2.0 --center-labels --center-local-loss-data --dataset CIFAR10 --device cuda:0 --divnorm-power 0.5 --dropout-p 0.05 --epoch-decrease-lr 300 350 450 475 --experiment vgg --final-lr 0.005 --final-sgd-momentum 0.95 --group-conv --group-fc --grouped-var-delta 1.0 --grouping-dim 64 --grouping-power 0.5 --hsic-estimate-mode plausible --hsic-gaussian-sigma-y 5.0 --hsic-gaussian-sigma-z 5.0 --hsic-kernel-y cossim --hsic-kernel-z cossim --local-cross-entropy-projection-size 2048 --local-loss-type cross-entropy --local-lr 0.001 --local-sgd-momentum 0.95 --lrelu-negative-slope 0.01 --mlp-layer-size 1024 --n-epochs 500 --nonlin selu --opt-lr-decrease 0.25 --optimizer-final SGD --optimizer-local SGD --training-mode test --validation-split 0.1 --vgg-conv-size-multiplier 1 --weight-decay-final 1e-06 --weight-decay-local 1e-07 >> "$log_filename"

echo 'Running cross-entropy_none_2'
log_filename="${logs_folder}/cross-entropy_none_2.out"
python3 experiments.py --batch-size 128 --bottleneck-gamma 2.0 --center-labels --center-local-loss-data --dataset CIFAR10 --device cuda:0 --divnorm-power 0.5 --dropout-p 0.05 --epoch-decrease-lr 300 350 450 475 --experiment vgg --final-lr 0.005 --final-sgd-momentum 0.95 --group-conv --group-fc --grouped-var-delta 1.0 --grouping-dim 64 --grouping-power 0.5 --hsic-estimate-mode plausible --hsic-gaussian-sigma-y 5.0 --hsic-gaussian-sigma-z 5.0 --hsic-kernel-y cossim --hsic-kernel-z cossim --local-cross-entropy-projection-size 2048 --local-loss-type cross-entropy --local-lr 0.001 --local-sgd-momentum 0.95 --lrelu-negative-slope 0.01 --mlp-layer-size 1024 --n-epochs 500 --nonlin selu --opt-lr-decrease 0.25 --optimizer-final SGD --optimizer-local SGD --training-mode test --validation-split 0.1 --vgg-conv-size-multiplier 2 --weight-decay-final 1e-06 --weight-decay-local 1e-07 >> "$log_filename"

echo 'Running sign_divnorm_1'
log_filename="${logs_folder}/sign_divnorm_1.out"
python3 experiments.py --backprop --backprop-batch-manhattan --batch-size 128 --bottleneck-gamma 2.0 --center-labels --center-local-loss-data --dataset CIFAR10 --device cuda:0 --divisive-norm-conv --divisive-norm-fc --divnorm-power 0.5 --dropout-p 0.05 --epoch-decrease-lr 300 350 450 475 --experiment vgg --final-lr 0.0005 --final-sgd-momentum 0.95 --grid-search-dummy --group-conv --group-fc --grouped-var-delta 1.0 --grouping-dim 64 --grouping-power 0.5 --hsic-estimate-mode plausible --hsic-gaussian-sigma-y 5.0 --hsic-gaussian-sigma-z 5.0 --hsic-kernel-y cossim --hsic-kernel-z cossim --local-cross-entropy-projection-size 2048 --local-loss-type HSIC --local-lr 0.001 --local-sgd-momentum 0.9 --lrelu-negative-slope 0.01 --mlp-layer-size 1024 --n-epochs 500 --nonlin selu --opt-lr-decrease 0.25 --optimizer-final SGD --optimizer-local SGD --sign-symmetry --training-mode test --validation-split 0.1 --vgg-conv-size-multiplier 1 --weight-decay-final 1e-06 --weight-decay-local 1e-07 >> "$log_filename"

echo 'Running cross-entropy-fa_divnorm_2'
log_filename="${logs_folder}/cross-entropy-fa_divnorm_2.out"
python3 experiments.py --batch-size 128 --bottleneck-gamma 2.0 --center-labels --center-local-loss-data --dataset CIFAR10 --device cuda:0 --divisive-norm-conv --divisive-norm-fc --divnorm-power 0.5 --dropout-p 0.05 --epoch-decrease-lr 300 350 450 475 --experiment vgg --final-lr 0.005 --final-sgd-momentum 0.95 --group-conv --group-fc --grouped-var-delta 1.0 --grouping-dim 128 --grouping-power 0.5 --hsic-estimate-mode plausible --hsic-gaussian-sigma-y 5.0 --hsic-gaussian-sigma-z 5.0 --hsic-kernel-y cossim --hsic-kernel-z cossim --local-cross-entropy-projection-size 2048 --local-loss-type cross-entropy-fa --local-lr 0.005 --local-sgd-momentum 0.95 --lrelu-negative-slope 0.01 --mlp-layer-size 1024 --n-epochs 500 --nonlin selu --opt-lr-decrease 0.25 --optimizer-final SGD --optimizer-local SGD --training-mode test --validation-split 0.1 --vgg-conv-size-multiplier 2 --weight-decay-final 1e-06 --weight-decay-local 1e-07 >> "$log_filename"

echo 'Running cross-entropy_divnorm_2'
log_filename="${logs_folder}/entropy_divnorm_2.out"
python3 experiments.py --batch-size 128 --bottleneck-gamma 2.0 --center-labels --center-local-loss-data --dataset CIFAR10 --device cuda:0 --divisive-norm-conv --divisive-norm-fc --divnorm-power 0.5 --dropout-p 0.05 --epoch-decrease-lr 300 350 450 475 --experiment vgg --final-lr 0.005 --final-sgd-momentum 0.95 --group-conv --group-fc --grouped-var-delta 1.0 --grouping-dim 64 --grouping-power 0.5 --hsic-estimate-mode plausible --hsic-gaussian-sigma-y 5.0 --hsic-gaussian-sigma-z 5.0 --hsic-kernel-y cossim --hsic-kernel-z cossim --local-cross-entropy-projection-size 2048 --local-loss-type cross-entropy --local-lr 0.005 --local-sgd-momentum 0.95 --lrelu-negative-slope 0.01 --mlp-layer-size 1024 --n-epochs 500 --nonlin selu --opt-lr-decrease 0.25 --optimizer-final SGD --optimizer-local SGD --training-mode test --validation-split 0.1 --vgg-conv-size-multiplier 2 --weight-decay-final 1e-06 --weight-decay-local 1e-07 >> "$log_filename"

echo 'Running fa_divnorm_1'
log_filename="${logs_folder}/fa_divnorm_1.out"
python3 experiments.py --backprop --batch-size 128 --bottleneck-gamma 2.0 --center-labels --center-local-loss-data --dataset CIFAR10 --device cuda:0 --divisive-norm-conv --divisive-norm-fc --divnorm-power 0.5 --dropout-p 0.05 --epoch-decrease-lr 300 350 450 475 --experiment vgg --feedback-alignment --final-lr 0.001 --final-sgd-momentum 0.95 --grid-search-dummy --group-conv --group-fc --grouped-var-delta 1.0 --grouping-dim 64 --grouping-power 0.5 --hsic-estimate-mode plausible --hsic-gaussian-sigma-y 5.0 --hsic-gaussian-sigma-z 5.0 --hsic-kernel-y cossim --hsic-kernel-z cossim --local-cross-entropy-projection-size 2048 --local-loss-type HSIC --local-lr 0.001 --local-sgd-momentum 0.9 --lrelu-negative-slope 0.01 --mlp-layer-size 1024 --n-epochs 500 --nonlin selu --opt-lr-decrease 0.25 --optimizer-final SGD --optimizer-local SGD --training-mode test --validation-split 0.1 --vgg-conv-size-multiplier 1 --weight-decay-final 1e-06 --weight-decay-local 0.0 >> "$log_filename"

echo 'Running sign_divnorm_2'
log_filename="${logs_folder}/sign_divnorm_2.out"
python3 experiments.py --backprop --backprop-batch-manhattan --batch-size 128 --bottleneck-gamma 2.0 --center-labels --center-local-loss-data --dataset CIFAR10 --device cuda:0 --divisive-norm-conv --divisive-norm-fc --divnorm-power 0.5 --dropout-p 0.05 --epoch-decrease-lr 300 350 450 475 --experiment vgg --final-lr 0.0005 --final-sgd-momentum 0.95 --grid-search-dummy --group-conv --group-fc --grouped-var-delta 1.0 --grouping-dim 128 --grouping-power 0.5 --hsic-estimate-mode plausible --hsic-gaussian-sigma-y 5.0 --hsic-gaussian-sigma-z 5.0 --hsic-kernel-y cossim --hsic-kernel-z cossim --local-cross-entropy-projection-size 2048 --local-loss-type HSIC --local-lr 0.001 --local-sgd-momentum 0.9 --lrelu-negative-slope 0.01 --mlp-layer-size 1024 --n-epochs 500 --nonlin selu --opt-lr-decrease 0.25 --optimizer-final SGD --optimizer-local SGD --sign-symmetry --training-mode test --validation-split 0.1 --vgg-conv-size-multiplier 2 --weight-decay-final 1e-06 --weight-decay-local 0.0 >> "$log_filename"

echo 'Running cross-entropy-fa_divnorm_1'
log_filename="${logs_folder}/cross-entropy-fa_divnorm_1.out"
python3 experiments.py --batch-size 128 --bottleneck-gamma 2.0 --center-labels --center-local-loss-data --dataset CIFAR10 --device cuda:0 --divisive-norm-conv --divisive-norm-fc --divnorm-power 0.5 --dropout-p 0.05 --epoch-decrease-lr 300 350 450 475 --experiment vgg --final-lr 0.005 --final-sgd-momentum 0.95 --group-conv --group-fc --grouped-var-delta 1.0 --grouping-dim 64 --grouping-power 0.5 --hsic-estimate-mode plausible --hsic-gaussian-sigma-y 5.0 --hsic-gaussian-sigma-z 5.0 --hsic-kernel-y cossim --hsic-kernel-z cossim --local-cross-entropy-projection-size 2048 --local-loss-type cross-entropy-fa --local-lr 0.005 --local-sgd-momentum 0.95 --lrelu-negative-slope 0.01 --mlp-layer-size 1024 --n-epochs 500 --nonlin selu --opt-lr-decrease 0.25 --optimizer-final SGD --optimizer-local SGD --training-mode test --validation-split 0.1 --vgg-conv-size-multiplier 1 --weight-decay-final 1e-06 --weight-decay-local 1e-07 >> "$log_filename"

echo 'Running cross-entropy_divnorm_1'
log_filename="${logs_folder}/cross-entropy_divnorm_1.out"
python3 experiments.py --batch-size 128 --bottleneck-gamma 2.0 --center-labels --center-local-loss-data --dataset CIFAR10 --device cuda:0 --divisive-norm-conv --divisive-norm-fc --divnorm-power 0.5 --dropout-p 0.05 --epoch-decrease-lr 300 350 450 475 --experiment vgg --final-lr 0.005 --final-sgd-momentum 0.95 --group-conv --group-fc --grouped-var-delta 1.0 --grouping-dim 64 --grouping-power 0.5 --hsic-estimate-mode plausible --hsic-gaussian-sigma-y 5.0 --hsic-gaussian-sigma-z 5.0 --hsic-kernel-y cossim --hsic-kernel-z cossim --local-cross-entropy-projection-size 2048 --local-loss-type cross-entropy --local-lr 0.005 --local-sgd-momentum 0.95 --lrelu-negative-slope 0.01 --mlp-layer-size 1024 --n-epochs 500 --nonlin selu --opt-lr-decrease 0.25 --optimizer-final SGD --optimizer-local SGD --training-mode test --validation-split 0.1 --vgg-conv-size-multiplier 1 --weight-decay-final 1e-06 --weight-decay-local 1e-07 >> "$log_filename"

echo 'Running fa_divnorm_2'
log_filename="${logs_folder}/fa_divnorm_2.out"
python3 experiments.py --backprop --batch-size 128 --bottleneck-gamma 2.0 --center-labels --center-local-loss-data --dataset CIFAR10 --device cuda:0 --divisive-norm-conv --divisive-norm-fc --divnorm-power 0.5 --dropout-p 0.05 --epoch-decrease-lr 300 350 450 475 --experiment vgg --feedback-alignment --final-lr 0.0005 --final-sgd-momentum 0.95 --grid-search-dummy --group-conv --group-fc --grouped-var-delta 1.0 --grouping-dim 128 --grouping-power 0.5 --hsic-estimate-mode plausible --hsic-gaussian-sigma-y 5.0 --hsic-gaussian-sigma-z 5.0 --hsic-kernel-y cossim --hsic-kernel-z cossim --local-cross-entropy-projection-size 2048 --local-loss-type HSIC --local-lr 0.001 --local-sgd-momentum 0.9 --lrelu-negative-slope 0.01 --mlp-layer-size 1024 --n-epochs 500 --nonlin selu --opt-lr-decrease 0.25 --optimizer-final SGD --optimizer-local SGD --training-mode test --validation-split 0.1 --vgg-conv-size-multiplier 2 --weight-decay-final 1e-06 --weight-decay-local 0.0 >> "$log_filename"

done

echo "Done"
