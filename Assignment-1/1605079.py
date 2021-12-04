import pprint as pp
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer

class Utility:

    @staticmethod
    def get_whitespace_count(df):
        columns = df.columns
        dict = {}
        for col in columns:
            dict[col] = df[col].str.isspace().sum() if df[col].dtype == 'object' else -1

        pp.pprint(dict)

    @staticmethod
    def get_value_counts(df, num_cols):
        for col in num_cols:
            print(df[col].value_counts())

    @staticmethod
    def minMaxScaler(df, transformable_columns, label):
        test = df.copy()

        if label in transformable_columns:
            transformable_columns.remove(label)

        for col in transformable_columns:
            print(col)
            test[col] = MinMaxScaler().fit_transform(test[[col]])

        return test

    @staticmethod
    def transformMinMaxScaler(df, transformable_columns, label):
        test = df.copy()

        if label in transformable_columns:
            transformable_columns.remove(label)

        for col in transformable_columns:
            print(col)
            test[col] = MinMaxScaler().fit_transform(test[[col]])

        return test

    @staticmethod
    def transformKBinsDiscretizer(df, transformable_columns, label, bins):

        if label in transformable_columns:
            transformable_columns.remove(label)

        for col in transformable_columns:
            est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
            df[col] = est.fit_transform(df[[col]])

def preprocess_churn_data(df, label):

    # create utility object
    prep_util = Utility()

    # # then we see information about dataset
    print(df.info())

    #print(df.dtypes)

    # drop the missing labels in the dataset
    print(len(df))

    df.dropna(axis=0, subset=[label])

    print(len(df))

    # drop the customer ID column in the dataset
    df.drop('customerID', axis=1, inplace=True)

    # converting the labels(y) to numeric labels
    label_encoder = preprocessing.LabelEncoder()
    df[label] = label_encoder.fit_transform(df[label])

    print("\nMissing values :  ", df.isnull().sum().values.sum())

    # get the whitespace  counts and remove them
    prep_util.get_whitespace_count(df)

    df = df.replace(r'^\s*$', np.NaN, regex=True)

    prep_util.get_whitespace_count(df)

    print("\nMissing values :  ", df.isnull().sum())

    # converting a single column to float
    # df[cols] = df[cols].apply(pd.to_numeric, errors='coerce') where cols are required columns we want to convert
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], downcast="float", errors='coerce')

    #print("\nBefore Missing values :  ", df.isnull().sum())

    # replacing the missing values with mean for total charges
    df['TotalCharges'].fillna(value=df['TotalCharges'].mean(), inplace=True)


    # first we separate the numerical and categorical features
    num_cols = df._get_numeric_data().columns
    columns = df.columns
    categorical_cols = list(set(columns) - set(num_cols))

    print('Numerical Cols: ', num_cols)
    print('All cols:', columns)
    print('categorical_cols:', categorical_cols)

    #print("\After Handling Missing values :  ", df.isnull().sum())

    # converting from categorical features to numerical features
    df = pd.get_dummies(df, columns=categorical_cols)

    num_cols_bin_cands = list(num_cols)
    num_cols_bin_cands.remove('SeniorCitizen')
    num_cols_bin_cands.remove('Churn')

    for col in num_cols_bin_cands:
        df[col] = StandardScaler().fit_transform(df[[col]])

    return df


def preprocess_adult_data(df, label):
    pass


def preprocess_credit_card_fraud_data(df, label):
    pass




df = pd.read_csv('./data/cust_churn.csv')

df = preprocess_churn_data(df, 'Churn')
print(df.head())
    