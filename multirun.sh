# for n_flag in {1..5..1}; do
#     CUDA_VISIBLE_DEVICES=1 python train_propensity.py --n_flag ${n_flag} --dataset coat --dropout 0.2 --dir ensemble  --seed $(($n_flag+40)) --sample_ratio -1 &
# done
# for epoch in 5 10 20; do
#     for label_smoothing in 0.02; do
#         for n_flag in {1..5..1}; do
#             CUDA_VISIBLE_DEVICES=1 python train_rec_with_ips.py --ps_epoch $epoch --n_flag ${n_flag} --label_smoothing ${label_smoothing} --dataset yahoo
#         done
#     done
# done