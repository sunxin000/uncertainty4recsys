gpus=(0 1 2 3 4 5 6)
datasets=('kuairand')

for dataset in "${datasets[@]}"; do 
    echo "benchmark" $dataset
    
    if [ -f "./new_results/${dataset}_bench.txt" ]; then
        rm "./new_results/${dataset}_bench.txt"
    fi
    
    for n_flag in {0..5..1}; do
        CUDA_VISIBLE_DEVICES=${gpus[n_flag]} python train_rec_benchmark.py \
        --dataset $dataset \
        --n_flag $n_flag \
        --seed $n_flag & 
    done

    wait

    python calculate_confidence_interval.py --name "${dataset}_bench.txt"
done
