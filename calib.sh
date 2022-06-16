# for dataset in yahoo coat
# do
#     for embedding_size in 64 96 128
#     do
#         for mlp_size in 16 32 64 96 128
#         do 
#             python calib.py --embedding_size ${embedding_size} --mlp_dim ${mlp_size} --dataset $dataset --sample_ratio 1
#         done
#     done
# done 


for epoch in {10..100..10}; do
    python propensity_analysis.py --epoch $epoch 
done