for epoch in {10..100..10}
do 
    python gen_propensity_ensemble.py --epoch $epoch 
done