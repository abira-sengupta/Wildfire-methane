# Residual plot - AdaBoost

![Example Image](images/Residuals-AdaBoost.png).

# Residual plot - Decision Tree

![Example Image](images/ResidualDT.jpg).


# Prediction Error - CatBoost

![Example Image](images/PredictionerrorCatBoost.jpg).

# SHAP - CataBoost

![Example Image](images/SHAPCatBoost.jpg).

# SHAP - AdaBoost

![Example Image](images/SHAP-AdaBoost.jpg).

# SHAP - Dependency plot
![Example Image](images/SHAPdependenceplot.jpg).

# PCA plot
![Example Image](images/pca_loadings_bar_chart.jpg).

# Table 1: An overview of the preprocessing information and PyCaret setup configuration for the regression experiment.

| Description                                   | Feature | Metric     |
|-----------------------------------------------|---------|------------|
| Wildfire combustion rate                      | crfire  | kg m⁻² s⁻¹ |
| Wildfire flux of black carbon                 | bcfire  | kg m⁻² s⁻¹ |
| Wildfire flux of carbon monoxide              | cofire  | kg m⁻² s⁻¹ |
| Wildfire flux of methane                      | CH4     | kg m⁻² s⁻¹ |
| Wildfire flux of non-methane hydrocarbons     | nmhcfire| kg m⁻² s⁻¹ |
| Wildfire flux of total particulate matter d < 2.5 µm (PM2.5) | tpmfire | kg m⁻² s⁻¹ |
| Wildfire fraction of area observed            | offire  | dimensionless |
| Wildfire fraction of area observed            | cffire  | kg m⁻² s⁻¹ |
| Wildfire radiative power                      | frpfire | W m⁻²      |
| Altitude of plume top                         | apt     | m          |

# Table 2: An overview of the preprocessing information and PyCaret setup configuration for the regression experiment

| Description                 | Value        |
|-----------------------------|--------------|
| Session id                  | 1234         |
| Target                      | ch4fire      |
| Target type                 | Regression   |
| Original data shape         | (862136, 4)  |
| Transformed data shape      | (862136, 4)  |
| Transformed train set shape | (603495, 4)  |
| Transformed test set shape  | (258641, 4)  |
| Numeric features            | 3            |
| Preprocess                  | True         |
| Imputation type             | simple       |
| Numeric imputation          | mean         |
| Fold Generator              | KFold        |
| Fold Number                 | 10           |
| CPU Jobs                    | -1           |

# SHAPLEY ADDITIVE EXPLANATION (SHAP)
In this work, we use SHAP (SHapley Additive exPlanations) plots to analyse the outputs of various machine learning algorithms. SHAP is based on game theory, 
where a scenario is modelled with $M$ players and a contribution function $v(S)$ represents the total expected payoff obtained by a subset of players $S$.

The Shapley value ensures a fair distribution of total gains among players. For a given player $j$, the Shapley value is defined as:

ϕ(v)j​=ϕj​=S⊆M∖{j}∑​∣M∣!∣S∣!(∣M∣−∣S∣−1)!​(v(S∪{j})−v(S)) ................  (1)

The difference term $\big(v(S \cup {j}) - v(S)\big)$ measures the additional contribution of player $j$ to the subset $S$. Thus, the Shapley value $\phi_j$ 
is the weighted mean of all possible additional contributions across subsets not containing $j$

Conditional Expectation Function

Shapley values can also be expressed using the conditional expectation of the model output:

fx​(S)=Ef​[f(X)∣do(XS​=xS​)] ................ (2)

Where:

> $S$: the set of features

> $X$: the random variable representing all $M$ input features

> $x$: the input vector for the current prediction

Key Properties of SHAP

Shapley values uniquely satisfy three properties:

Local Accuracy (Efficiency)
The sum of feature attributions equals the model’s prediction:

f(x)=ϕ0​(f)+i=1∑M​ϕi​(f,x) .................. (3)

Here, $\phi_0(f)$ is the expected model output $E[f(Z)]$, and the sum of $\phi_i(f, x)$ matches $f(x)$.

Consistency (Monotonicity)
If a model changes so that a feature’s contribution increases (or remains constant), its attribution should not decrease. For two models $f$ and $f'$:

fx′​(S)−fx′​(S∖i)≥fx​(S)−fx​(S∖i)⇒ϕi​(f′,x)≥ϕi​(f,x) .....................(4)

Missingness (Null Effects)
A feature that does not affect the model output receives a Shapley value of zero:

fx​(S∪i)=fx​(S)⇒ϕi​(f,x)=0 .....................(5)

# Correlation - map

![Example Image](images/correlation.png).

# Table 3: PyCaret modelling

| Model   | Description                        | MAE    | MSE    | RMSE   | R²      | RMSLE  | MAPE   | TT (Sec) |
|---------|------------------------------------|--------|--------|--------|---------|--------|--------|----------|
| catboost| CatBoost Regressor                 | 0.0327 | 0.0047 | 0.0682 | 0.1029  | 0.0575 | 8.5502 | 14.1200  |
| lightgbm| Light Gradient Boosting Machine    | 0.0323 | 0.0047 | 0.0683 | 0.1023  | 0.0575 | 8.4318 | 136.1240 |
| gbr     | Gradient Boosting Regressor        | 0.0329 | 0.0047 | 0.0687 | 0.0895  | 0.0579 | 9.1552 | 5.5720   |
| rf      | Random Forest Regressor            | 0.0342 | 0.0048 | 0.0691 | 0.0788  | 0.0587 | 9.2734 | 16.5240  |
| et      | Extra Trees Regressor              | 0.0345 | 0.0048 | 0.0695 | 0.0694  | 0.0590 | 9.4238 | 4.9490   |
| xgboost | Extreme Gradient Boosting          | 0.0336 | 0.0048 | 0.0696 | 0.0670  | 0.0589 | 8.8548 | 0.3240   |
| ridge   | Ridge Regression                   | 0.0349 | 0.0050 | 0.0706 | 0.0411  | 0.0596 | 11.0725| 0.0770   |
| br      | Bayesian Ridge                     | 0.0349 | 0.0050 | 0.0706 | 0.0411  | 0.0596 | 11.0727| 0.0460   |
| lr      | Linear Regression                  | 0.0349 | 0.0050 | 0.0706 | 0.0411  | 0.0596 | 11.0725| 0.6800   |
| lar     | Least Angle Regression             | 0.0353 | 0.0050 | 0.0708 | 0.0344  | 0.0598 | 11.3288| 0.0770   |
| omp     | Orthogonal Matching Pursuit        | 0.0353 | 0.0051 | 0.0712 | 0.0240  | 0.0602 | 11.7457| 0.0700   |
| lasso   | Lasso Regression                   | 0.0364 | 0.0052 | 0.0721 | -0.0001 | 0.0610 | 13.0236| 0.0530   |
| en      | Elastic Net                        | 0.0364 | 0.0052 | 0.0721 | -0.0001 | 0.0610 | 13.0236| 0.1290   |
| llar    | Lasso Least Angle Regression       | 0.0364 | 0.0052 | 0.0721 | -0.0001 | 0.0610 | 13.0236| 0.0900   |
| dummy   | Dummy Regressor                    | 0.0364 | 0.0052 | 0.0721 | -0.0001 | 0.0610 | 13.0236| 0.1320   |
| knn     | K Neighbors Regressor              | 0.0349 | 0.0055 | 0.0739 | -0.0530 | 0.0633 | 8.4704 | 0.2630   |
| huber   | Huber Regressor                    | 0.0260 | 0.0056 | 0.0745 | -0.0707 | 0.0632 | 2.5873 | 0.2010   |
| ada     | AdaBoost Regressor                 | 0.0596 | 0.0071 | 0.0840 | -0.3607 | 0.0744 | 24.2473| 0.7080   |
| dt      | Decision Tree Regressor            | 0.0419 | 0.0099 | 0.0995 | -0.9110 | 0.0834 | 9.4509 | 0.3550   |
| par     | Passive Aggressive Regressor       | 0.0706 | 0.0103 | 0.0982 | -1.0337 | 0.0841 | 30.6262| 0.1000   |
