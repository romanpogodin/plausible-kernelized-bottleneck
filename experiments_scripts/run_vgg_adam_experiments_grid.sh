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

# new experiments with feedback alignment, sign symmetry and layer-wise classification

python3 --batch-size 128 --bottleneck-gamma 2.0 --center-labels --center-local-loss-data --dataset CIFAR10 --device cuda:0 --divnorm-power 0.5 --dropout-p 0.05 --epoch-decrease-lr 300 350 450 475 --experiment vgg --final-lr 0.005 --final-sgd-momentum 0.95 --group-conv --group-fc --grouped-var-delta 1.0 --grouping-dim 64 --grouping-power 0.5 --hsic-estimate-mode plausible --hsic-gaussian-sigma-y 5.0 --hsic-gaussian-sigma-z 5.0 --hsic-kernel-y cossim --hsic-kernel-z cossim --local-cross-entropy-projection-size 2048 --local-loss-type cross-entropy --local-lr 0.001 --local-sgd-momentum 0.95 --lrelu-negative-slope 0.01 --mlp-layer-size 1024 --n-epochs 500 --nonlin selu --opt-lr-decrease 0.25 --optimizer-final SGD --optimizer-local SGD --results-filename "logs/cross-entropy-none-1/final-$i" --save-results --training-mode test --validation-split 0.1 --vgg-conv-size-multiplier 1 --weight-decay-final 1e-06 --weight-decay-local 1e-07 >> "$log_filename"

python3 --batch-size 128 --bottleneck-gamma 2.0 --center-labels --center-local-loss-data --dataset CIFAR10 --device cuda:0 --divnorm-power 0.5 --dropout-p 0.05 --epoch-decrease-lr 300 350 450 475 --experiment vgg --final-lr 0.005 --final-sgd-momentum 0.95 --group-conv --group-fc --grouped-var-delta 1.0 --grouping-dim 64 --grouping-power 0.5 --hsic-estimate-mode plausible --hsic-gaussian-sigma-y 5.0 --hsic-gaussian-sigma-z 5.0 --hsic-kernel-y cossim --hsic-kernel-z cossim --local-cross-entropy-projection-size 2048 --local-loss-type cross-entropy --local-lr 0.001 --local-sgd-momentum 0.95 --lrelu-negative-slope 0.01 --mlp-layer-size 1024 --n-epochs 500 --nonlin selu --opt-lr-decrease 0.25 --optimizer-final SGD --optimizer-local SGD --results-filename "logs/cross-entropy-none-2/final-$i" --save-results --training-mode test --validation-split 0.1 --vgg-conv-size-multiplier 2 --weight-decay-final 1e-06 --weight-decay-local 1e-07 >> "$log_filename"

python3 --backprop --batch-norm --batch-size 128 --bottleneck-gamma 2.0 --center-labels --center-local-loss-data --dataset CIFAR10 --device cuda:0 --divnorm-power 0.2 --dropout-p 0.05 --epoch-decrease-lr 300 350 450 475 --experiment vgg --feedback-alignment --final-lr 0.0005 --final-sgd-momentum 0.95 --grid-search-dummy --group-conv --group-fc --grouped-var-delta 1.0 --grouping-dim 64 --grouping-power 0.5 --hsic-estimate-mode plausible --hsic-gaussian-sigma-y 5.0 --hsic-gaussian-sigma-z 5.0 --hsic-kernel-y cossim --hsic-kernel-z cossim --local-lr 0.001 --local-sgd-momentum 0.9 --lrelu-negative-slope 0.01 --mlp-layer-size 1024 --n-epochs 500 --nonlin lrelu --opt-lr-decrease 0.25 --optimizer-final AdamW --optimizer-local SGD --results-filename final --training-mode test --validation-split 0.1 --vgg-conv-size-multiplier 1 --weight-decay-final 1e-06 --weight-decay-local 0.0 >> "$log_filename"

python3 --batch-norm --batch-size 128 --bottleneck-gamma 2.0 --center-labels --center-local-loss-data --dataset CIFAR10 --device cuda:0 --divnorm-power 0.5 --dropout-p 0.05 --epoch-decrease-lr 300 350 450 475 --experiment vgg --final-lr 0.005 --final-sgd-momentum 0.95 --group-conv --group-fc --grouped-var-delta 1.0 --grouping-dim 64 --grouping-power 0.5 --hsic-estimate-mode plausible --hsic-gaussian-sigma-y 5.0 --hsic-gaussian-sigma-z 5.0 --hsic-kernel-y cossim --hsic-kernel-z cossim --local-cross-entropy-projection-size 2048 --local-loss-type cross-entropy-fa --local-lr 0.0005 --local-sgd-momentum 0.95 --lrelu-negative-slope 0.01 --mlp-layer-size 1024 --n-epochs 500 --nonlin lrelu --opt-lr-decrease 0.25 --optimizer-final AdamW --optimizer-local AdamW --results-filename "logs/cross-entropy-fa-bn-2/final-$i" --save-results --training-mode test --validation-split 0.1 --vgg-conv-size-multiplier 2 --weight-decay-final 1e-06 --weight-decay-local 1e-07 >> "$log_filename"

python3 --batch-norm --batch-size 128 --bottleneck-gamma 2.0 --center-labels --center-local-loss-data --dataset CIFAR10 --device cuda:0 --divnorm-power 0.5 --dropout-p 0.05 --epoch-decrease-lr 300 350 450 475 --experiment vgg --final-lr 0.005 --final-sgd-momentum 0.95 --group-conv --group-fc --grouped-var-delta 1.0 --grouping-dim 64 --grouping-power 0.5 --hsic-estimate-mode plausible --hsic-gaussian-sigma-y 5.0 --hsic-gaussian-sigma-z 5.0 --hsic-kernel-y cossim --hsic-kernel-z cossim --local-cross-entropy-projection-size 2048 --local-loss-type cross-entropy --local-lr 0.0005 --local-sgd-momentum 0.95 --lrelu-negative-slope 0.01 --mlp-layer-size 1024 --n-epochs 500 --nonlin lrelu --opt-lr-decrease 0.25 --optimizer-final AdamW --optimizer-local AdamW --results-filename final --training-mode test --validation-split 0.1 --vgg-conv-size-multiplier 1 --weight-decay-final 1e-06 --weight-decay-local 1e-07 >> "$log_filename"

python3 --backprop --batch-norm --batch-size 128 --bottleneck-gamma 2.0 --center-labels --center-local-loss-data --dataset CIFAR10 --device cuda:0 --divnorm-power 0.5 --dropout-p 0.05 --epoch-decrease-lr 300 350 450 475 --experiment vgg --final-lr 0.0005 --final-sgd-momentum 0.95 --grid-search-dummy --group-conv --group-fc --grouped-var-delta 1.0 --grouping-dim 64 --grouping-power 0.5 --hsic-estimate-mode plausible --hsic-gaussian-sigma-y 5.0 --hsic-gaussian-sigma-z 5.0 --hsic-kernel-y cossim --hsic-kernel-z cossim --local-lr 0.001 --local-sgd-momentum 0.9 --lrelu-negative-slope 0.01 --mlp-layer-size 1024 --n-epochs 500 --nonlin lrelu --opt-lr-decrease 0.25 --optimizer-final AdamW --optimizer-local SGD --results-filename final --sign-symmetry --training-mode test --validation-split 0.1 --vgg-conv-size-multiplier 1 --weight-decay-final 1e-06 --weight-decay-local 0.0 >> "$log_filename"

python3 --backprop --batch-norm --batch-size 128 --bottleneck-gamma 2.0 --center-labels --center-local-loss-data --dataset CIFAR10 --device cuda:0 --divnorm-power 0.5 --dropout-p 0.05 --epoch-decrease-lr 300 350 450 475 --experiment vgg --feedback-alignment --final-lr 0.0005 --final-sgd-momentum 0.95 --grid-search-dummy --group-conv --group-fc --grouped-var-delta 1.0 --grouping-dim 64 --grouping-power 0.5 --hsic-estimate-mode plausible --hsic-gaussian-sigma-y 5.0 --hsic-gaussian-sigma-z 5.0 --hsic-kernel-y cossim --hsic-kernel-z cossim --local-cross-entropy-projection-size 2048 --local-loss-type HSIC --local-lr 0.001 --local-sgd-momentum 0.9 --lrelu-negative-slope 0.01 --mlp-layer-size 1024 --n-epochs 500 --nonlin lrelu --opt-lr-decrease 0.25 --optimizer-final AdamW --optimizer-local SGD --results-filename "logs/fa-bn-2/final-$i" --save-results --training-mode test --validation-split 0.1 --vgg-conv-size-multiplier 2 --weight-decay-final 1e-06 --weight-decay-local 0.0 >> "$log_filename"

python3 --batch-norm --batch-size 128 --bottleneck-gamma 2.0 --center-labels --center-local-loss-data --dataset CIFAR10 --device cuda:0 --divnorm-power 0.5 --dropout-p 0.05 --epoch-decrease-lr 300 350 450 475 --experiment vgg --final-lr 0.005 --final-sgd-momentum 0.95 --group-conv --group-fc --grouped-var-delta 1.0 --grouping-dim 64 --grouping-power 0.5 --hsic-estimate-mode plausible --hsic-gaussian-sigma-y 5.0 --hsic-gaussian-sigma-z 5.0 --hsic-kernel-y cossim --hsic-kernel-z cossim --local-cross-entropy-projection-size 2048 --local-loss-type cross-entropy-fa --local-lr 0.001 --local-sgd-momentum 0.95 --lrelu-negative-slope 0.01 --mlp-layer-size 1024 --n-epochs 500 --nonlin lrelu --opt-lr-decrease 0.25 --optimizer-final AdamW --optimizer-local AdamW --results-filename final --training-mode test --validation-split 0.1 --vgg-conv-size-multiplier 1 --weight-decay-final 1e-06 --weight-decay-local 1e-07 >> "$log_filename"

python3 --batch-norm --batch-size 128 --bottleneck-gamma 2.0 --center-labels --center-local-loss-data --dataset CIFAR10 --device cuda:0 --divnorm-power 0.5 --dropout-p 0.05 --epoch-decrease-lr 300 350 450 475 --experiment vgg --final-lr 0.005 --final-sgd-momentum 0.95 --group-conv --group-fc --grouped-var-delta 1.0 --grouping-dim 64 --grouping-power 0.5 --hsic-estimate-mode plausible --hsic-gaussian-sigma-y 5.0 --hsic-gaussian-sigma-z 5.0 --hsic-kernel-y cossim --hsic-kernel-z cossim --local-cross-entropy-projection-size 2048 --local-loss-type cross-entropy --local-lr 0.0005 --local-sgd-momentum 0.95 --lrelu-negative-slope 0.01 --mlp-layer-size 1024 --n-epochs 500 --nonlin lrelu --opt-lr-decrease 0.25 --optimizer-final AdamW --optimizer-local AdamW --results-filename "logs/cross-entropy-bn-2/final-$i" --save-results --training-mode test --validation-split 0.1 --vgg-conv-size-multiplier 2 --weight-decay-final 1e-06 --weight-decay-local 1e-07 >> "$log_filename"

python3 --backprop --batch-norm --batch-size 128 --bottleneck-gamma 2.0 --center-labels --center-local-loss-data --dataset CIFAR10 --device cuda:0 --divnorm-power 0.5 --dropout-p 0.05 --epoch-decrease-lr 300 350 450 475 --experiment vgg --final-lr 0.0005 --final-sgd-momentum 0.95 --grid-search-dummy --group-conv --group-fc --grouped-var-delta 1.0 --grouping-dim 64 --grouping-power 0.5 --hsic-estimate-mode plausible --hsic-gaussian-sigma-y 5.0 --hsic-gaussian-sigma-z 5.0 --hsic-kernel-y cossim --hsic-kernel-z cossim --local-cross-entropy-projection-size 2048 --local-loss-type HSIC --local-lr 0.001 --local-sgd-momentum 0.9 --lrelu-negative-slope 0.01 --mlp-layer-size 1024 --n-epochs 500 --nonlin lrelu --opt-lr-decrease 0.25 --optimizer-final AdamW --optimizer-local SGD --results-filename final --sign-symmetry --training-mode test --validation-split 0.1 --vgg-conv-size-multiplier 2 --weight-decay-final 1e-06 --weight-decay-local 0.0 >> "$log_filename"

done

echo "Done"
