for label_smoothing in 0.01 0.02 0.03 0.04 0.05 0.08 0.10 0.20
do 
    echo $label_smoothing
    for n_flag in {1..5..1}; do
        python train_rec_with_ips.py --dataset yahoo --n_flag $n_flag --label_smoothing $label_smoothing --ps_epoch 20 
    done
done 