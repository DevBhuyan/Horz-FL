# Cost-Efficient Feature Selection for Horizontal Federated Learning


# Quickstart:
```bash
clone git repo
cd ~/Horz-FL
pip install -r requirements.txt
python one_round.py
```

Recommended Python Version: `3.9`

# Driver Files:
1. **one_round.py**
	This is the main script for Horizontal FL, Inside this you will get options to run different FL methods across different datasets and compile results.
2. **model_size_vs_accuracy.py**
	[EXPERIMENTAL] This script will inspect the size of a federated forest with respect to the returned accuracy and compute the optimal size of trees which gives maximum accuracy for different datasets across different FS algorithms.
3. **sota_driver.py**
	This file contains our implementation of the FSHFL algorithm proposed by Zhang et. Al.
4. **generate_synthetic_data.py**
	Running this file will generate a synthetic dataset as used in our non-iid experiments.
5. **federated_ridge_regression.py**
	This additional script was written to compute the results for the two regression datasets
6. **fed_mrmr.py**
	This file contains our Python implementation of the Fed-mRMR algorithm proposed by Hermo et. Al.
	
# Other helper files:
 - calc_MI.py
 - Cluster_kmeans.py
 - results_utils.py
 - run_iid.py
 - run_noniid.py
 - SMOTE.py
 - train_model.py
 - data_prep.py
 - MLP_helpers.py
 - sota.py
 - FAR_based_outlier_detection.py
 - Fed_MLP.py
 - ff.py
 - ff_helpers.py
 - frhc.py
 - global_feature_select.py
 - global_update.py
 - helpers.py
 - horz_data_divn.py
 - local_feature_select.py
 - local_update.py
 - normalize.py
 - NSGA_2.py
 - options.py
 - preprocesing.py

The datasets directory can be downloaded from [here](https://drive.google.com/file/d/1OgmWSRQkSaRYkNr9uju_qejz-lrPRgkf/view?usp=sharing) ~ 163.7 MB

EDIT: Newer datasets are larger, it is recommended that you check the dataset_sources.txt to download all the datasets.
