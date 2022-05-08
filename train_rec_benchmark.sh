for weight_decay in 0.1 0.01 0.001 0.0001; do
    for lr in 0.01 0.001; do
        CUDA_VISIBLE_DEVICES=2 python train_rec_benchmark.py --dataset coat --weight_decay ${weight_decay} --lr ${lr} &
    done
done