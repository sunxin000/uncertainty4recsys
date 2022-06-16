for n_flag in {1..5..1}; do
    CUDA_VISIBLE_DEVICES=0 python train_rec_with_MRDR.py --dataset yahoo --n_flag $n_flag --label_smoothing 0.0 --ps_epoch 20 &
done