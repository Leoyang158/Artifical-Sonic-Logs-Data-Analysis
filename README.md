# Artifical-Sonic-Logs-Data-Analysis
Basic Challenge: "The challenge will be a regression exercise to generate artificial sonic logs perfectly from the least number of available logs" 

#Purpose
The purpose of this challenge is to benchmark and discover new approaches that may lead to a class of more reliable production forecasts than traditional approaches such as Arps, modified hyperbolic, Duong and logistic DCA models.

#Goal: 
Build a data-driven model to predict both compressional travel time (DT) and shear travel time (DTS) logs using least number of logs / inputs from the logging-while-drilling dataset

#Problem Description:
Type: Supervised Learning Problem 
Formulation: Predict sonic log through a non linear mapping between features
Inputs: Available well logs (Porosity, Gammar Ray, Resistivity & Compressional slowness)
Output: DT and DTS

#Data Pre-processing Steps
1. Identify outliers with the 13/5 Interquartile Rangers rule
2. Fill Nan values with median in the test set. Though that will cause noise, compared with building other predictive model to fill the gap, the negative effect is less. 
3. Based on statistical methods, most of the outliers has been removed. We then use isolation forest to gently remove some outliers near the boundary to ensure the stability and standardization of the dataset. 
4. Draw a distribution plot to visualize the processed data set and observe whether each feature conforms to the normal distribution. 
5. Use Pearson and Spearman correlation method to select features and rank the relevance of features to prepare for modeling.

#Methodology and improvement 
Try Decision Tree, Random Forest, and xgbooster gradient boosting, and compared their corresponding accuracy. Xgbooster gradient Boosting > Random Tree > Decision Tree 
Hyperparameter Optimization (The Best could be 98! ) 
Cross-validation + Comparison of accuracy curves between test group and training group --- The purpose is to determine whether the forecast curve has a trend of overfitting or underfitting 
Improvement: In the process of data preprocessing, the most accurate way is to map variables to high-dimensional space. The advantage is that all the information of the original data is completely retained, without considering missing values, without considering issues such as linear inseparability.

Visualization
![image](https://user-images.githubusercontent.com/62164871/111835711-47d3df80-88c3-11eb-86d2-c5497e047b6c.png)
![image](https://user-images.githubusercontent.com/62164871/111835723-4d312a00-88c3-11eb-9a12-c004fef6e0be.png)

#Programming Language: Python
#Packages: Numpy, Pandas, Sklearning #xgboost #random_forest
