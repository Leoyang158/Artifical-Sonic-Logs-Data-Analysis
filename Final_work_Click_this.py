# import package
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import xgboost


def read_file(filename, sheetname):
    df = pd.read_excel(filename, sheetname, skiprows=24450, nrows=10968)
    df.columns = ["Depth", "CALI", "DRHO", "DT", "DTS", "GR", "NPHI", "PEF", "RHOB", "ROP", "RT"]
    df.values.reshape(10968, 11)
    df = df.set_index("Depth")
    df.replace(-999.25, np.nan, inplace=True)
    return df

def Depth_Graph(data_set):
    array = ["Depth", "CALI", "DRHO", "DT", "DTS", "GR", "NPHI", "PEF", "RHOB", "ROP", "RT"]
    for i in range(1, len(array)):
        data_set.plot.scatter(x=array[i], y='Depth')
        plt.show()

def clean_outlier(data_set):
    array = ["Depth", "CALI", "DRHO", "DT", "DTS", "GR", "NPHI", "PEF", "RHOB", "ROP", "RT"]
    for i in range(1, len(array)):
        # sns.boxplot(x=data_set[array[i]])
        # plt.show() #Show the range of each data set
        q3 = np.nanquantile(data_set[array[i]], .75)
        q1 = np.nanquantile(data_set[array[i]], .25)
        IQR = q3 - q1
        # Using the IQR to clean out the outlier
        lower_bound = q1 - (13 / 5) * IQR
        upper_bound = q3 + (13 / 5) * IQR
        data_set.loc[data_set[array[i]] < lower_bound, array[i]] = -999.25
        data_set.loc[data_set[array[i]] > upper_bound, array[i]] = -999.25
        data_set.replace(-999.25, np.nan, inplace=True)
        # sns.boxplot(x=data_set[array[i]])
        # plt.show() #Show the range of each data set after cleaning out the outlier

def fill_median(data_set):
    array = ["Depth", "CALI", "DRHO", "DT", "DTS", "GR", "NPHI", "PEF", "RHOB", "ROP", "RT"]
    for i in range(1, len(array)):
        data_set[array[i]].replace(np.nan, data_set[array[i]].median(), inplace=True)

def hist_graph(data_set):
    fig, ax = plt.subplots(figsize=(15, 15))
    data_set.hist(ax=ax)
    plt.show()

def read_deployment(filename, sheetname, variable, add_list):
    df = pd.read_excel(filename, sheet_name=sheetname, skiprows=3, nrows=183)
    df.columns = ['Depth', 'CALI', 'DRHO', 'GR', 'NPHI', 'PEF', 'RHOB', 'ROP', 'RT', 'EMPTY', 'DT', 'DTS']
    if variable == 'DT':
        df.drop(['EMPTY', 'DT', 'DTS'], axis=1, inplace=True)
        df.replace(-999.25, np.nan, inplace=True)
        df.values.reshape(183, 9)
        df = df.set_index('Depth')
        df.values.reshape(183, 8)
    elif variable == 'DTS':
        df.drop(['EMPTY', 'DTS'], axis=1, inplace=True)
        df['DT'] = add_list
        df.replace(-999.25, np.nan, inplace=True)
        df.values.reshape(183, 10)
        df = df.set_index('Depth')
        df.values.reshape(183, 9)
    return df

def predict_to_excel(predict_dt, predict_dts, filename, sheet_name, new_file):
    ExcelDataInPandasDataFrame = pd.read_excel(filename, sheet_name=sheet_name, skiprows=3, nrows=183)
    ExcelDataInPandasDataFrame.columns = ['Depth', 'CALI', 'DRHO', 'GR', 'NPHI', 'PEF', 'RHOB', 'ROP', 'RT',
                                          'EMPTY', 'DT', 'DTS']
    ExcelDataInPandasDataFrame['DT'] = predict_dt
    ExcelDataInPandasDataFrame["DTS"] = predict_dts
    Predict_file = new_file
    ExcelDataInPandasDataFrame.to_excel(Predict_file, sheet_name=sheet_name, startrow=3, index=False)

class Machine_Learning_Algorithm(object):
    def __init__(self, data_set, variable, list):
        self.data_set = data_set
        self.select_element = variable
        if variable == 'DT':
            self.features = ['CALI', 'DRHO', 'GR', 'NPHI', 'PEF', 'RHOB', 'ROP', 'RT']
        elif variable == 'DTS':
            self.features = ['CALI', 'DRHO', 'GR', 'NPHI', 'PEF', 'RHOB', 'ROP', 'RT', 'DT']
        self.x = data_set.iloc[:, data_set.columns != variable]
        self.y = data_set.iloc[:, data_set.columns == variable]
        self.deployment_list = list
        self.deployment_depth = []

    def Check_Data_distribution(self):
        fig, ax = plt.subplots(figsize=(15, 15))
        self.data_set.hist(ax=ax)
        plt.show()

    def XGBoost(self, filename, sheetname):  # don't use this function yet
        # try XGBoost
        if(self.select_element == 'DT'): # if DT, then drop DTS
            self.x.drop(['DTS'], axis=1, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.25, random_state=0)
        # Scale data
        scaler = RobustScaler(quantile_range=(2.5, 97.5))
        X_train_array = scaler.fit_transform(X_train)
        X_test_array = scaler.transform(X_test)
        xgb = xgboost.XGBRegressor(objective='reg:squarederror')
        xgb.fit(X_train_array, y_train)
        y_test_pred = xgb.predict(X_test_array)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(y_test, y_test_pred)
        # Show True DT and Predicted DT
        xlabel_string = 'True ' + self.select_element
        ylabel_string = 'Predicted ' + self.select_element
        ax.set(xlabel=xlabel_string, ylabel=ylabel_string)
        plt.show()
        print('Accuracy',r2_score(y_test, y_test_pred))
        if self.select_element == 'DT':
            addlist = []
        elif self.select_element == 'DTS':
            addlist = self.deployment_list
        df = read_deployment(filename, sheetname, self.select_element, addlist)
        self.deployment_depth = df.index.tolist()
        # Using model to test the predicted data
        model = xgb
        df_test = df[self.features].copy()
        if self.select_element == 'DT':
            df_test.values.reshape(183, 8)
        elif self.select_element == 'DTS':
            df_test.values.reshape(183,9)
        df_test_final = scaler.fit_transform(df_test)
        # make prediction for DT/DTS using the trained model
        y_test = model.predict(df_test_final)
        # print(y_test)
        # print(len(y_test))
        return y_test

    def remove_outlier_upfront(self):
        iforest = IsolationForest(n_estimators=200, contamination=0.5, random_state=0)
        iforest = iforest.fit(self.data_set)
        iforest_pred = iforest.predict(self.data_set)
        mask_in = iforest_pred == 1
        data_in = self.data_set[mask_in]
        fig, ax = plt.subplots(figsize=(15, 15))
        data_in.hist(ax=ax)
        plt.show()

    def cor_decision(self):
        x = data_set.iloc[:, data_set.columns != self.select_element]
        y = data_set.iloc[:, data_set.columns == self.select_element]
        df_X = self.data_set[self.features]
        df_y = self.data_set[[self.select_element]]

        df_corr = pd.concat([df_X, df_y], axis=1, ignore_index=True)
        df_corr.columns = self.features + [self.select_element]
        df_corr.head()
        corr_pearson = df_corr.corr(method='pearson')
        corr_spearman = df_corr.corr(method='spearman')

        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        sns.heatmap(corr_pearson, xticklabels=corr_pearson.columns, yticklabels=corr_pearson.columns,
                    center=0, vmin=-1, vmax=1, cmap='bwr', ax=ax[0])
        sns.heatmap(corr_spearman, xticklabels=corr_spearman.columns, yticklabels=corr_spearman.columns,
                    center=0, vmin=-1, vmax=1, cmap='bwr', ax=ax[1])

        ax[0].set(title='Pearson')
        ax[1].set(title='Spearman')
        plt.show()
        corr_spearman[self.select_element].abs().sort_values(ascending=False)
        top_features = corr_spearman[self.select_element].abs().sort_values(ascending=False)[:10].index.to_list().copy()
        print("Top features: \n")
        print(top_features)

    def predict_vs_actual(self, data_set_copy, predicted_array):
        array_depth = data_set_copy.index.tolist()
        array_actual = data_set_copy[self.select_element].tolist()
        array_predict_fill = data_set_copy[self.select_element].tolist()
        j = 0
        for i in self.deployment_depth:
            index = array_depth.index(i)
            array_predict_fill[index] = predicted_array[j]
            j += 1

        fig,ax = plt.subplots(1,2,figsize=(30,30),sharey=True)
        ax[0].plot(array_actual,array_depth)
        ax[0].invert_yaxis()
        title = self.select_element + ' vs Depth Raw Data'
        ax[0].set(title= title)
        ax[0].grid()
        ax[1].plot(array_predict_fill, array_depth)
        title_2 = self.select_element + ' vs Depth Prediction'
        ax[1].set(title = title_2)
        ax[1].grid()
        plt.suptitle('Comparison')
        plt.show()


if __name__ == '__main__':
    filename = "Problem 2 Dataset (Clean).xlsx"
    sheet_name = "Training-Testing"
    data_set = read_file(filename, sheet_name)
    data_set_2 = data_set.copy()
    data_set_3 = data_set.copy()
    clean_outlier(data_set)
    fill_median(data_set)
    # hist_graph(data_set) (optional)
    ###DT
    p1 = Machine_Learning_Algorithm(data_set, 'DT', list = [])
    p1.remove_outlier_upfront()
    p1.Check_Data_distribution()
    p1.cor_decision()
    Predict_DT = p1.XGBoost("Problem 2 Dataset (Clean).xlsx", "Deployment")
    Predict_DT = Predict_DT.tolist()
    p1.predict_vs_actual(data_set_2, Predict_DT)
    ############
    ###DTS
    p2 = Machine_Learning_Algorithm(data_set, 'DTS', list = Predict_DT)
    p2.remove_outlier_upfront()
    p2.Check_Data_distribution()
    p2.cor_decision()
    Predict_DTS = p2.XGBoost("Problem 2 Dataset (Clean).xlsx", "Deployment")
    p2.predict_vs_actual(data_set_3, Predict_DTS)
    ########### Output Predicted Data
    predict_to_excel(Predict_DT, Predict_DTS, "Problem 2 Dataset (Clean).xlsx", "Deployment", "New_data.xlsx")


