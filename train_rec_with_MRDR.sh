label_smoothing=0.0
ps_epoch=100
CUDA=1
for dir in raw MC_Dropout Deep_Ensembles Platt_Scaling; do 
    for data in coat yahoo; do
        rm "./results/${data}_MRDR_${dir}.txt"
        for n_flag in {1..10..1}; do
            CUDA_VISIBLE_DEVICES=$CUDA python train_rec_with_MRDR.py --dataset $data --dir $dir --n_flag $n_flag \
            --label_smoothing $label_smoothing --ps_epoch $ps_epoch &
        done
        wait
        python calculate_confidence_interval.py --name "${data}_MRDR_${dir}.txt"
    done
done