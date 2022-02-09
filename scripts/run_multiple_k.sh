METHOD=$1
CONFIG=$2
RES_DIR=$3

for DATASET in 'default_credit' 'biodeg' 'e_coli' 'energy' 'german_credit' 'image' 'letter' 'magic' 'parkinson' 'skillcraft' 'statlog' 'steel' 'taiwan_credit' 'wine_quality'
do
    for SEED in 0 1 2 3 4
    do
        python pareto.py --method $METHOD --config $CONFIG --dataset $DATASET --train_percent 0.5 --seed $SEED --result_folder $RES_DIR
    done
    echo "DONE" $METHOD "on" $DATASET
done
