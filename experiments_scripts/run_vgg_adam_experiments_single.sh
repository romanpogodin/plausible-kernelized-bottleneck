#!/bin/bash
args="python3 experiments.py "
args+="--device cuda:0 --training-mode test "  # validation --validation_split 0.1
args+="--vgg-conv-size-multiplier $x "
args+="--center-labels "
args+="--n-epochs 500 --epoch-decrease-lr 300 350 450 475 "
args+="--opt-lr-decrease 0.25 "
args+="--batch-size 128 "
args+="--final-lr $lro --local-lr $lrb --weight-decay-local 1e-7 --weight-decay-final 1e-6 --dropout-p 0.05 "
args+="--optimizer-final AdamW --optimizer-local AdamW "
args+="--nonlin lrelu  --batch-norm "
args+="--center-local-loss-data --group-conv --group-fc "
args+="--grouped-var-delta 1.0 "
args+="--grouping-dim $dim "
args+="--hsic-estimate-mode plausible "
args+="--local-sgd-momentum 0.95 --final-sgd-momentum 0.95 "
#args+="--spatial_dropout "

if [ "$mode" == 'divnorm' ]; then
  args+="--divisive-norm-conv --divisive-norm-fc --divnorm-power 0.2 --grouping-power 0.8 "  # grouping-power is 0.5 by default
fi

if [ "$rule" == 'gauss' ]; then
  args+="--hsic-kernel-z gaussian --hsic-gaussian-sigma-z 5 "  # y is cossim by default
fi

if [ "$rule" == 'backprop' ]; then
  args+="--backprop "
fi

echo "Running as $args" >> "$log_filename"
eval $args >> "$log_filename"

echo "Done"
exit
