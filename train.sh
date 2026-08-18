#! /usr/bin/bash
python train.py \
--data-dir /home/ubuntu/act/convert_out_stride2/convert_out_stride2 \
--ckpt-dir /home/ubuntu/act/train_test_pwrLimit400.bak \
--num-epochs 400 \
--batch-size 16 \
--chunk-size 10 \
--action-space cartesian_abs \
--camera-names chest top wrist_2 \
--lr 5e-6 \
--kl-weight 10 \
--grad-clip 1.0 \
--num-workers 6 \
# --resume-from /home/ubuntu/act/train_test_pwrLimit250.bak/policy_last.ckpt