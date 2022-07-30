# for weight_decay in 0.1 0.01 0.001 0.0001; do
#     for lr in 0.01 0.001; do
#         CUDA_VISIBLE_DEVICES=2 python train_rec_benchmark.py --dataset coat --weight_decay ${weight_decay} --lr ${lr} &
#     done
# done
dataset='coat'
for n_flag in {1..10..1}; do
    python train_rec_benchmark.py --dataset $dataset --n_flag $n_flag --seed $n_flag &
done

wait

python calculate_confidence_interval.py --name "${dataset}_bench.txt"
