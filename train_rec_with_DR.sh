dir='ls'
data='yahoo'
label_smoothing=0.02
ps_epoch=20
CUDA=1

for n_flag in {1..10..1}; do
    CUDA_VISIBLE_DEVICES=$CUDA python train_rec_with_DR.py --dataset $data --dir $dir --n_flag $n_flag \
    --label_smoothing $label_smoothing --ps_epoch $ps_epoch &
done

wait

python calculate_confidence_interval.py --name "${data}_DR_${dir}.txt"