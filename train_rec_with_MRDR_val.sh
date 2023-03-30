dir='ls'
data='yahoo'
CUDA=3
basebone='neumf'
for n_flag in {1..10..1}; do
    CUDA_VISIBLE_DEVICES=$CUDA python train_rec_with_MRDR_val.py \
    --dataset $data --n_flag $n_flag --dir $dir  --basebone $basebone &
done

wait 

python calculate_confidence_interval.py --name "${data}_MRDR_${dir}_${basebone}_val.txt"