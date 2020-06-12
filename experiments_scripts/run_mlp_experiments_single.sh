#!/bin/bash
args="python3 experiments.py "
args+="--device cuda:0 --training-mode test "  # validation --validation_split 0.1
args+="--experiment mlp --dataset $dataset "
args+="--mlp-layer-size 1024 "
args+="--center-labels "
args+="--n-epochs 100 --epoch-decrease-lr 50 75 90 "
args+="--opt-lr-decrease 0.25 "
args+="--batch-size 256 "
args+="--final-lr $lro --local-lr $lrb --weight-decay-local 1e-7 --weight-decay-final 1e-6 --dropout-p 0.01 "
args+="--optimizer-final SGD --optimizer-local SGD "
args+="--nonlin lrelu "
args+="--hsic-estimate-mode plausible "
args+="--local-sgd-momentum 0.95 --final-sgd-momentum 0.95 "

if [ "$mode" != 'none' ]; then
    args+="--center-local-loss-data --group-fc "
    args+="--grouped-var-delta 1.0 "
    args+="--grouping-dim $dim "
fi

if [ "$mode" == 'divnorm' ]; then
  args+="--divisive-norm-fc --divnorm-power 0.2 --grouping-power 0.8 "  # grouping-power is 0.5 by default
fi

if [ "$rule" == 'gauss' ]; then
  args+="--hsic-kernel-z gaussian --hsic-gaussian-sigma-z 5.0 "  # y is cossim by default
fi

if [ "$rule" == 'backprop' ]; then
  args+="--backprop "
fi

echo "Running as $args" >> "$log_filename"
eval $args >> "$log_filename"

echo "Done"
exit
