## Distributionally Robust Fair Principal Components via Geodesic Descents


### Reference
* Code for `CFPCA` and preprocessed data: https://github.com/molfat66/FairML
* Code for `FairPCA`: https://github.com/samirasamadi/Fair-PCA

### Generate Figure 1.
* run command: `$ python test.py`

### Generate Figure 2.
* First, generate results with `.pkl` format by running code:
`$ sh scripts/ablation_study.sh`
* Then, gather results and plot:
`$ python functions <dataset name> --res_dir <results folder>`

### Generate Figure 3.
* First, generate results by running code:
`$ sh scripts/run_multiple_k.sh <method> <path to config> <result folder>`
* Then, gather results and plot:
`$ python agg_multiple_k.py <dataset name> --res_dir <result folder> --img_dir <output image dir>`

### Generate Table 1.
* First, gererate results by running code: 
`$ sh run_all.sh <method> <path to config> <result folder>`
* Then, gather results and log:
`$ python agg_results.sh <dataset name/space split dataset names> --res_dir <result folder>`
