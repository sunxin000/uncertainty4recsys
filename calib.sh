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


# for epoch in {10..100..10}; do
#     python propensity_analysis.py --epoch $epoch 
# done

# python calib.py --path propensity/raw/coat_epoch_100_1_dropout_0.2_label_smoothing_0.0.pt --dir raw --dataset coat
# python calib.py --path propensity/raw/yahoo_epoch_20_1_dropout_0.2_label_smoothing_0.0.pt --dir raw --dataset yahoo
# python calib.py --path propensity/dropout/coat.pt --dir dropout --dataset coat
# python calib.py --path propensity/dropout/yahoo.pt --dir dropout --dataset yahoo
python calib.py --path propensity/ensemble/coat_seed.pt --dir seed-ensemble --dataset coat
python calib.py --path propensity/ensemble/yahoo_seed.pt --dir seed-ensemble --dataset yahoo
# python calib.py --path propensity/label_smoothing/coat_epoch_100_1_dropout_0.2_label_smoothing_0.1.pt --dir ls --dataset coat 
# python calib.py --path propensity/label_smoothing/yahoo_epoch_20_1_dropout_0.2_label_smoothing_0.02.pt --dir ls --dataset yahoo
# python calib.py --path propensity/ls+do/coat_1_10.pt --dir ls+drop --dataset coat
# python calib.py --path propensity/ls+do/yahoo_1_10.pt --dir ls+drop --dataset yahoo
python calib.py --path propensity/ls+ensemble/coat_seed.pt --dir seed-ls+ensemble --dataset coat
python calib.py --path propensity/ls+ensemble/yahoo_seed.pt --dir seed-ls+ensemble --dataset yahoo
