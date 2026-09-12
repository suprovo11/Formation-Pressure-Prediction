## Formation Pressure Prediction

This repository contains a leakage-controlled machine learning workflow for **formation pressure prediction** using drilling-log data. The target variable is **FPress (Formation Pressure, psi)**, based on a dataset containing 2,774 observations and 28 variables.

The implementation is designed to provide a more realistic evaluation of model generalization by preserving the sequential/depth structure of drilling-log data and preventing information leakage during preprocessing and feature selection.

### Methodology

The workflow includes:

* Data cleaning and removal of constant and duplicate features.
* Depth-based ordering of drilling-log observations.
* Training-only Pearson correlation feature selection (`|r| > 0.4`).
* Fold-specific Savitzky–Golay smoothing.
* Depth-ordered blocked cross-validation using an expanding-window strategy.
* Comparison of multiple regression algorithms:

  * Random Forest
  * Gradient Boosting
  * Multi-Layer Perceptron (MLP)
  * Support Vector Regression (SVR)
  * Ridge Regression
* Comparison against:

  * Mean baseline
  * Persistence baseline
* Evaluation using:

  * R²
  * RMSE
  * MAE
  * RPD
* Explicit comparison of training and testing performance to identify overfitting.
* Generation of CSV result tables and PNG performance visualizations.

The implementation specifically addresses leakage that can occur when feature selection and smoothing are performed before train/test separation. It also replaces random splitting with contiguous depth-ordered splits because adjacent drilling-log observations can be strongly autocorrelated.

### Model Evaluation

Five blocked folds are used by default. The final evaluation reports the mean and standard deviation of testing performance across folds, providing an indication of model stability rather than relying on a single potentially favorable split.

The principal evaluation metrics are:

| Metric   | Interpretation                                      |
| -------- | --------------------------------------------------- |
| **R²**   | Explained variance; higher is better                |
| **RMSE** | Prediction error in psi; lower is better            |
| **MAE**  | Mean absolute prediction error; lower is better     |
| **RPD**  | Ratio of performance to deviation; higher is better |

### Reference Benchmark

The original study associated with this task used **Principal Component Regression (PCR)** and reported:

* **R² = 0.78**
* **RPD = 0.922**

These values are retained as a reference benchmark; however, direct numerical comparison should be made cautiously because the present implementation uses a different, more conservative depth-ordered validation strategy.

### Output

Running the script creates a `results_formation_pressure/` directory containing:

```text
results_formation_pressure/
├── aggregated_results.csv
├── per_fold_results.csv
├── train_vs_test_r2.png
├── test_r2_by_fold.png
└── rmse_rpd.png
```

The generated results provide both aggregated model performance and fold-level performance for reproducible analysis.

### Reproducibility

The workflow uses a fixed random seed (`42`) and defines the principal configuration parameters in the script, including the correlation threshold, number of blocked folds, and final hold-out fraction.

### Citation

If you use this implementation or the underlying formation-pressure prediction methodology in academic work, please cite the original study:

```bibtex
@article{formation_pressure_prediction,
  author  = {Salman Shakib Suprova},
  year    = {2026},
  title   = {Formation Pressure Prediction},
  note    = {Principal Component Regression-based formation pressure prediction study}
}
```

