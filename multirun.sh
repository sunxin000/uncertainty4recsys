for n_flag in {1..5..1}; do
    CUDA_VISIBLE_DEVICES=3 python train_propensity.py --n_flag ${n_flag} --dataset coat --dropout 0.2 --dir ensemble  --seed $(($n_flag+40)) &
done