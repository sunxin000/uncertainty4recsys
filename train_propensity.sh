# for dataset in coat 
# do
#     for embedding_size in 64 96 128
#     do
#         for mlp_size in 16 32 64 96 128
#         do 
#             python train_propensity.py --embedding_size ${embedding_size} --mlp_dim ${mlp_size} --dataset $dataset --sample_ratio 1
#         done
#     done
# done 

# for label_smoothing in 0.01 0.02 0.03 0.04 0.05 0.08 0.10 0.20
# do 
#     python train_propensity.py --label_smoothing $label_smoothing --dataset yahoo --gen_ps &
# done

for seed in {40..49..1}
do 
    CUDA_VISIBLE_DEVICES=1  python train_propensity.py --label_smoothing 0.02 --dataset yahoo --seed $seed --n_flag $seed --dir ensemble &
done