# for label_smoothing in 0.01 0.02 0.03 0.04 0.05 0.08 0.10 0.20
# do 
#     echo $label_smoothing
#     for n_flag in {1..5..1}; do
#         python train_rec_with_ips.py --dataset yahoo --n_flag $n_flag --label_smoothing $label_smoothing --ps_epoch 20 
#     done
# done 
gpus=(2 3 4 5 6 7)
dirs=('raw' 'MC_Dropout' 'Platt_Scaling' 'Deep_Ensembles' 'dfl')
# dirs=('Platt_Scaling')
datasets=('kuairand')
for dir in "${dirs[@]}"; do
    for dataset in "${datasets[@]}"; do 
        echo $dir $dataset
        path="propensity/${dir}/${dataset}.pt"
        
        if [ -f "./results/${dataset}_ips_${dir}.txt" ]; then
            rm "./results/${dataset}_ips_${dir}.txt"
        fi
        for n_flag in {0..5..1}; do
            CUDA_VISIBLE_DEVICES=${gpus[n_flag]}$ python train_rec_with_ips.py --dir $dir --dataset $dataset \
            --n_flag $n_flag --seed $n_flag \
            --path $path  & 
        done

        wait

        python calculate_confidence_interval.py --name "${dataset}_ips_${dir}.txt"
    done
done