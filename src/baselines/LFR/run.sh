coeffRange=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.5 4.0 4.5 5.0 6.0 7.0 8.0 9.0 10.0 12.0 14.0 16.0 18.0 20.0)
seeds=(509 510 511 512 513)
datasets=(german compas adult)


for coeff in "${coeffRange[@]}"; do
    for dataset in "${datasets[@]}"; do
        for seed in "${seeds[@]}"; do
            python lfr.py $dataset 1 $coeff $seed
            python lfr.py $dataset $coeff 1 $seed
        done
    done
done