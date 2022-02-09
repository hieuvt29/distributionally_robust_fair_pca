for DATASET in 'adult_income' 'biodeg' 'e_coli' 'energy' 'german_credit' 'image' 'letter' 'magic' \
                'parkinson' 'pima' 'recidivism' 'skillcraft' 'statlog' 'steel' 'taiwan_credit' 'wine_quality'
do 
    python functions.py $DATASET 
done