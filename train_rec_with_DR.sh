for n_flag in {1..5..1}; do
    CUDA_VISIBLE_DEVICES=0 python train_rec_with_DR.py --dataset yahoo --n_flag $n_flag --label_smoothing 0.02 --ps_epoch 20 &
done