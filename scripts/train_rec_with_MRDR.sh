label_smoothing=0.0
ps_epoch=100
gpus=(0 1 3 4 5 6 7)
dirs=('raw' 'MC_Dropout' 'Platt_Scaling' 'Deep_Ensembles' 'dfl')
datasets=('coat' 'yahoo' 'kuairand')

for dir in "${dirs[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo $dir $dataset
        path="propensity/${dir}/${dataset}.pt"
        
        if [ -f "./new_results/${dataset}_MRDR_${dir}.txt" ]; then
            rm "./new_results/${dataset}_MRDR_${dir}.txt"
        fi
        
        for n_flag in {0..5..1}; do
            CUDA_VISIBLE_DEVICES=${gpus[n_flag]}$ python train_rec_with_MRDR.py \
            --dir $dir \
            --dataset $dataset \
            --n_flag $n_flag \
            --path $path &
        done
        
        wait
        
        python calculate_confidence_interval.py --name "${dataset}_MRDR_${dir}.txt"
    done
done