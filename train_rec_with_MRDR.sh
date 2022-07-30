dir='ls'
data='coat'
for n_flag in {1..10..1}; do
    CUDA_VISIBLE_DEVICES=1 python train_rec_with_MRDR.py --dataset $data --n_flag $n_flag --dir $dir --label_smoothing 0.1 --ps_epoch 100 &
done

wait 

python calculate_confidence_interval.py --name "MF_${data}_MRDR_${dir}.txt"