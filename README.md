## README file for Horz-FL-main

# Quickstart:
```bash
git clone https://github.com/DevBhuyan/Horz-FL.git
cd ~/Horz-FL
pip install -r requirements.txt
python ten_rounds.py
```

# Driver Files:
1. **ten_rounds.py**
	This is the main script for Horizontal FL, Inside this you will get options to run different FL methods across different datasets and compile results.
2. **ten_rounds_benchmarker.py**
	This is a similar implementation of ten_rounds but for No-Feature Selection. This is meant to generate the benchmark metrics (accuracy, precision, recall) upto which the FS returned data must be able to perform.
3. **model_size_vs_accuracy.py**
	[EXPERIMENTAL] This script will inspect the size of a federated forest with respect to the returned accuracy and compute the optimal size of trees which gives maximum accuracy for different datasets across different FS algorithms.
4. **sota_driver.py**
	This file contains our implementation of the FSHFL algorithm proposed by Zhang et. Al.
5. **dataset_reduction_task.py**
	This experimental script will run the different FS algorithms and return the reduced datasets for human inspection.
6. **communication_cost.py**
	[EXPERIMENTAL] This script will inspect the number of communication rounds of the FedAvg algorithm with respect to the returned accuracy and compute the optimal number of iterations which gives maximum accuracy for different datasets across different FS algorithms.
7. **for_plots.py**
	[EXPERIMENTAL] This script will show the plots of FCMI and aFFMI scores for various datasets.
8. **optimize_fedavg.py**
	PyTorch implementation of communication_cost.py, highly optimized, runs on command-line arguments and is much quicker than its older counterpart
9. **Fed_MoFS.py** and **Fed_FiS.py** are the standalone implementations of our proposed methods.

# Other helper files:
- anova.py
- calc_MI.py
- Cluster_kmeans.py
- data_prep.py
- FAR_based_outlier_detection.py
- Fed_MLP.py
- ff_helpers.py
- ff.py
- frhc.py
- global_feature_select.py
- global_update.py
- horz_data_divn.py
- local_feature_select.py
- local_update.py
- MLP_helpers.py
- normalize.py
- NSGA_2.py
- optimize_fedavg.py
- options.py
- preprocesing.py
- randomforest.py
- results_utils.py
- SMOTE.py
- sota.py
- train_model.py

The datasets directory can be downloaded from [here](https://drive.google.com/file/d/1OgmWSRQkSaRYkNr9uju_qejz-lrPRgkf/view?usp=sharing) ~ 163.7 MB
