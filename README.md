# MFIA
A Physics-Informed Multivariate Ensemble Framework for Seismic Fragility of Shield Tunnels with Heteroscedastic and Non-Gaussian Demands
(1) DoSE+Dynamic.py – used for DoSE sampling and rotation/scaling of ground motions;
(2) mes_training.py – used for training the MES surrogate model;
(3) mes_generate_predictions.py – used for batch prediction with the trained MES model, including generation of 10,000 predicted samples (mean, standard deviation, and confidence intervals);
(4) calculate_residual_correlation.py – used for calculating the residual correlation coefficients on the training set for subsequent copula-based fragility analysis;
(5) probabilistic_modeling.py – used for computing standardized residuals, constructing the ABKDE model, and building the Gaussian copula model based on residual correlation;
(6) MES_TASK_2.csv – the dataset containing the results of 1,000 nonlinear time-history analyses.
