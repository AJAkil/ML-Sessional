# %%
import pprint as pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import argparse
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer


class InformationGain:

    def __init__(self, df, num_cols, label):
        self.df = df
        self.num_cols = num_cols
        self.label = label
        self.original_columns = self.df.columns

    def get_final_column_list(self, num_of_features):
        self.cont_to_bins_pipeline()  # careful here!
        gain_dict = {col: self.calculate_gain(col) for col in list(set(self.df.columns) - {self.label})}
        sorted_gain_dict = {k: v for k, v in sorted(gain_dict.items(), key=lambda item: item[1])}
        print('Sorted gain dict:', sorted_gain_dict)
        cols_to_drop = list(sorted_gain_dict.keys())[:len(sorted_gain_dict) - num_of_features]
        final_cols = list(set(self.original_columns) - set(cols_to_drop))
        # print('Final Cols are: ==>')
        # print(final_cols)

        return final_cols

    def calculate_gain(self, attribute):
        if self.label != attribute:
            p = len(self.df[self.df[self.label] == 1])
            df_size = len(self.df)  # p + n
            data_entropy = self.calculate_entropy(p / df_size)
            # print('Data entropy is', data_entropy)

            attribute_remainder = self.calculate_remainder(attribute)
            return data_entropy - attribute_remainder

    def calculate_remainder(self, attribute):
        unique_vals, num_of_unique = np.unique(self.df[attribute], return_counts=True)

        remainder_sum = 0
        for index, attrib_val in enumerate(unique_vals):
            # choosing the rows equal to the unique value in the attribute
            filtered_df = self.df.where(self.df[attribute] == attrib_val).dropna()

            # calculating number of positive classed rows for the given attribute's unique value
            pk = len(filtered_df[filtered_df[self.label] == 1])
            nk = len(filtered_df[filtered_df[self.label] == 0])

            # sanity check
            assert nk == len(filtered_df) - pk

            prob = (pk + nk) / (len(self.df))
            attr_entropy = self.calculate_entropy(pk / (pk + nk))
            remainder_sum += prob * attr_entropy

        return remainder_sum

    @staticmethod
    def calculate_entropy(q):
        if q > 0:
            return -1 * (q * np.log2(q) + (1 - q) * np.log2(1 - q))
        return 0

    def convert_cont_to_bins(self, old_col_name):
        min = self.df[old_col_name].min()
        median = self.df[old_col_name].median()

        if min != median:
            # print(old_col_name)
            self.df[old_col_name + '_'] = pd.qcut(self.df[old_col_name], q=4, labels=['q1', 'q2', 'q3', 'q4'])
            self.df.drop(columns=[old_col_name], inplace=True)
            self.df.rename(columns={old_col_name + '_': old_col_name}, inplace=True)

    def cont_to_bins_pipeline(self):
        for col in list(set(self.num_cols) - set(self.label)):
            self.convert_cont_to_bins(col)


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
    def transformStandardScaler(df, transformable_columns, label):
        test = df.copy()

        if label in transformable_columns:
            transformable_columns.remove(label)

        test[transformable_columns] = StandardScaler().fit_transform(test[transformable_columns])

        return test

    @staticmethod
    def transformMinMaxScaler(df, transformable_columns, label):
        test = df.copy()

        if label in transformable_columns:
            transformable_columns.remove(label)

        test[transformable_columns] = MinMaxScaler().fit_transform(test[transformable_columns])

        return test

    @staticmethod
    def transformKBinsDiscretizer(df, transformable_columns, label, bins):

        if label in transformable_columns:
            transformable_columns.remove(label)

        for col in transformable_columns:
            est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
            df[col] = est.fit_transform(df[[col]])

    @staticmethod
    def get_binary_col_count(df, columns):
        return [col for col in columns if len(df[col].value_counts()) == 2]

    def get_all_cols(self, df):
        columns = list(df.columns)
        columns_with_nan = df.columns[df.isna().any()].tolist()
        num_cols = list(df._get_numeric_data().columns)
        cat_cols = list(set(columns) - set(num_cols))
        cat_cols_with_nan = set(columns_with_nan) - set(num_cols)
        num_cols_with_nan = set(columns_with_nan) - set(cat_cols)
        binary_cols = self.get_binary_col_count(df, columns)

        return {'columns': columns,
                'columns_with_nan': columns_with_nan,
                'num_cols': num_cols,
                'cat_cols': cat_cols,
                'cat_cols_with_nan': cat_cols_with_nan,
                'num_cols_with_nan': num_cols_with_nan,
                'binary_cols': binary_cols}


class MetricCalculator:
    def __init__(self, y_real, y_pred) -> None:
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.y_real = y_real
        self.y_pred = y_pred
        self.num_rows = len(y_pred)

    def calculate_cf_matrix_fields(self):

        self.y_real[self.y_real == 0] = -1

        for index in range(self.num_rows):
            if self.y_real[index] == 1 and self.y_pred[index] == 1:
                self.TP += 1
            if self.y_pred[index] == 1 and self.y_real[index] == -1:
                self.FP += 1
            if self.y_real[index] == -1 and self.y_pred[index] == -1:
                self.TN += 1
            if self.y_pred[index] == -1 and self.y_real[index] == 1:
                self.FN += 1

    def get_cf_field(self):
        return self.TP, self.TN, self.FP, self.FN

    def calculate_all_metric(self):
        self.calculate_cf_matrix_fields()
        self.calculate_accuracy()
        self.calculate_recall()
        self.calculate_specificity()
        self.calculate_precision()
        self.calculate_false_discovery_rate()
        self.calculate_f1_score()

        print(f'TP: {self.TP}')
        print(f'TN: {self.TN}')
        print(f'FP: {self.FP}')
        print(f'FN: {self.FN}')

        print(f'Accuracy: {self.calculate_accuracy()}')
        print(f'Recall: {self.calculate_recall()}')
        print(f'Specificity: {self.calculate_specificity()}')
        print(f'Precision: {self.calculate_precision()}')
        print(f'False Discovery Rate: {self.calculate_false_discovery_rate()}')
        print(f'F1 score: {self.calculate_f1_score()}')

    def calculate_accuracy(self):
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)

    def calculate_precision(self):
        return self.TP / (self.TP + self.FP)

    def calculate_recall(self):
        return self.TP / (self.TP + self.FN)

    def calculate_specificity(self):
        return self.TN / (self.TN + self.FP)

    def calculate_false_discovery_rate(self):
        return self.FP / (self.FP + self.TP)

    def calculate_f1_score(self):
        return (2 * self.TP) / (2 * self.TP + self.FP + self.FN)


def preprocess_churn_data(df, label, num_of_features):
    util = Utility()

    # print(len(df))

    df.dropna(axis=0, subset=[label])

    # print(len(df))

    # drop the customer ID column in the dataset
    df.drop('customerID', axis=1, inplace=True)

    # converting the labels(y) to numeric labels
    # print(df.head())
    label_encoder = preprocessing.LabelEncoder()
    df[label] = label_encoder.fit_transform(df[label])
    # print(df.head())

    # print("\nMissing values :  ", df.isnull().sum().values.sum())

    # get the whitespace  counts and remove them
    # util.get_whitespace_count(df)

    df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.NaN, regex=True)

    # util.get_whitespace_count(df)

    # print("\nMissing values :  ", df.isnull().sum())

    # converting a single column to float
    # df[cols] = df[cols].apply(pd.to_numeric, errors='coerce') where cols are required columns we want to convert
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], downcast="float", errors='coerce')

    # print("\nBefore Missing values :  ", df.isnull().sum())

    # replacing the missing values with mean for total charges
    df['TotalCharges'].fillna(value=df['TotalCharges'].mean(), inplace=True)

    # print("\nAfter Missing values :  ", df.isnull().sum())

    if num_of_features != -1:
        # get all columns
        col_name_dict = util.get_all_cols(df)

        # gain computations
        temp_df = df.copy()
        gainFilter = InformationGain(temp_df, col_name_dict['num_cols'], label)
        final_cols = gainFilter.get_final_column_list(num_of_features=num_of_features)

        # print('Final Cols are: ==>')
        # print(final_cols)
        df = df[final_cols]

    # get all columns
    col_name_dict = util.get_all_cols(df)

    # removing categorical columns with mode( most frequent value)
    for cat_col in col_name_dict['cat_cols_with_nan']:
        df[cat_col].fillna(value=df[cat_col].mode()[0], inplace=True)

    # print(df.isnull().sum())

    # removing numerical columns with mean value
    for num_col in col_name_dict['num_cols_with_nan']:
        df[num_col].fillna(value=df[num_col].mean(), inplace=True)

    # print(df.isnull().sum())

    # one hot encoding the categorical cols
    df = pd.get_dummies(df, columns=list(set(col_name_dict['cat_cols']) - set(col_name_dict['binary_cols'])))

    # label encoding the binary cols
    for col in list(set(col_name_dict['binary_cols']) - set(label)):
        # print('Label Encoding: ', col)
        label_encoder = preprocessing.LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col])

    if len(list(set(col_name_dict['num_cols']) - set(col_name_dict['binary_cols']))) != 0:
        df = util.transformStandardScaler(df, list(set(col_name_dict['num_cols']) - set(col_name_dict['binary_cols'])),
                                          label)

    # changing the lables from 0,1 to -1,1
    df[label] = df[label].replace([0], -1)

    df.reset_index(inplace=True, drop=True)

    return df


def preprocess_adult_data(df, label):
    pass


def preprocess_credit_card_fraud_data(df, label):
    pass


class LogisticRegression:
    def __init__(self, learning_rate, max_iter, test_size, early_stop_error, decay) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.cost_history = []
        self.learning_rate_history = []
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.test_size = test_size
        self.label = None
        self.early_stop_error = early_stop_error
        self.decay = decay

    def split_given_train_test_df(self, train_df, test_df, label):
        self.label = label

        y_train = train_df[label]
        x_train = train_df.drop(label, axis=1)

        self.y_train = np.array(y_train).reshape(y_train.shape[0], 1)
        self.x_train = np.array(x_train)

        y_test = test_df[label]
        x_test = test_df.drop(label, axis=1)

        self.y_test = np.array(y_test).reshape(y_test.shape[0], 1)
        self.x_test = np.array(x_test)

    def split_dataset(self, df, label):

        self.label = label
        y = df[label]
        X = df.drop(label, axis=1)

        y = np.array(y).reshape(y.shape[0], 1)
        X = np.array(X)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size,
                                                                                random_state=111)

    def fit(self, is_constant_lr, no_curve, calculate_metric_on_train):
        # df[label] = df[label].replace([-1],0)
        n_samples, n_features = self.x_train.shape

        # initializing weights
        # self.uniform_weight_initializer(n_features)
        self.zero_initializer(n_features)

        # initializing cost history
        self.cost_history = np.zeros((self.max_iter, 1))
        self.learning_rate_history = np.zeros((self.max_iter, 1))

        for epoch in range(self.max_iter):
            # print(X.shape, self.weights.shape)
            # print('Epoch: ', epoch)

            if not is_constant_lr:
                # adjusting learning rate in each epoch
                self.learning_rate_scheduler(epoch)
                self.learning_rate_history[epoch] = self.learning_rate

            h_w = np.tanh(np.matmul(self.x_train, self.weights))
            X_T = np.transpose(self.x_train)

            assert h_w.shape == self.y_train.shape

            y_h_w = self.y_train - h_w
            tan_der = 1 - np.square(h_w)

            assert y_h_w.shape == tan_der.shape

            self.weights = self.weights + (2 * self.learning_rate) * (1 / n_samples) * np.matmul(X_T, np.multiply(y_h_w,
                                                                                                                  tan_der))
            cost = self.calculate_mse_cost(y_h_w)
            self.cost_history[epoch] = cost

            if cost < self.early_stop_error:
                print(cost)
                print('Stopping Training since the error is less than 0.5')
                break

        if not no_curve:
            self.plot_cost_vs_iteration()
            self.plot_learning_rate_curve()

        # calculate metrics on training set
        if calculate_metric_on_train:
            y_pred = self.predict(self.x_train)
            metric = MetricCalculator(self.y_train, y_pred)
            metric.calculate_all_metric()

    def predict(self, x):
        y_pred = np.tanh(x @ self.weights)

        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = -1

        return y_pred

    def uniform_weight_initializer(self, num_features):
        self.weights = np.random.uniform(low=0, high=1, size=num_features).reshape(num_features, 1)

    def xavier_initialization(self, num_features):
        self.weights = np.full((num_features, 1), np.sqrt(1 / num_features))  # Xavier Initialization

    def zero_initializer(self, num_features):
        self.weights = np.zeros((num_features, 1))

    def generate_metric(self):
        y_pred = self.predict(self.x_test)
        metric = MetricCalculator(self.y_test, y_pred)
        metric.calculate_all_metric()

    @staticmethod
    def calculate_mse_cost(y_h_w):
        return np.mean(np.square(y_h_w))

    def plot_cost_vs_iteration(self):

        print(min(self.cost_history))
        plt.figure()
        plt.plot(range(self.max_iter), self.cost_history)
        plt.title('Cost Function Convergence Curve')
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost")
        plt.show()

    def plot_learning_rate_curve(self):
        plt.figure()
        plt.plot(range(self.max_iter), self.learning_rate_history)
        plt.title('Learning Rate Decay Curve')
        plt.xlabel("Number of Iterations")
        plt.ylabel("Learning Rate")
        plt.show()

    def learning_rate_scheduler(self, epoch):
        self.learning_rate *= (1. / (1. + self.decay * epoch))


class Adaboost:

    def __init__(self, num_of_learner, test_size):
        self.test_size = test_size
        self.y_test = None
        self.y_train = None
        self.x_test = None
        self.x_train = None
        self.num_of_learner = num_of_learner
        self.W = []
        self.h = []
        self.z = []
        self.label = None
        self.train_df = None
        self.test_df = None

    def split_dataset(self, df, label):
        self.label = label

        self.train_df, self.test_df = train_test_split(df, test_size=self.test_size, random_state=111)

    def split_dataset_given_tran_test(self, train_df, test_df, label):
        self.label = label
        self.train_df = train_df
        self.test_df = test_df

    def convert_df_to_np(self, df):
        y = df[self.label]
        X = df.drop(self.label, axis=1)

        y = np.array(y).reshape(y.shape[0], 1)
        X = np.array(X)

        return X, y

    def fit(self, base_learner_max_iter, error, learning_rate, decay):

        # setting the weights
        N = self.train_df.shape[0]
        self.W = [1.0 / N] * N
        self.h = [None] * self.num_of_learner
        self.z = [None] * self.num_of_learner

        # keeping a copy of the original train dataframe
        original_train_df = self.train_df.copy()

        # getting numpy arrays for original train dataframe
        X_original_train_df, y_original_train_df = self.convert_df_to_np(original_train_df)

        for k in range(self.num_of_learner):
            # print('Boosting Round: ', k)
            lgr_learner = LogisticRegression(learning_rate, base_learner_max_iter, 0.2, error, decay)

            resampled_df = original_train_df.sample(n=N, weights=self.W, replace=True, random_state=111)

            # convert resampled_df to fit to logistic regression
            X, y = self.convert_df_to_np(resampled_df)

            lgr_learner.x_train = X
            lgr_learner.y_train = y

            # train the weak learner
            lgr_learner.fit(is_constant_lr=True, no_curve=True, calculate_metric_on_train=False)

            # storing the weak learner
            self.h[k] = lgr_learner

            error = 0.0

            # getting the prediction of the weak learner for the original training data
            y_pred = self.h[k].predict(X_original_train_df)

            for i in range(N):
                if y_pred[i] != y_original_train_df[i]:
                    error += self.W[i]

            if error > 0.5:
                self.z[k] = 0
                continue

            for i in range(N):
                if y_pred[i] == y_original_train_df[i]:
                    self.W[i] = (self.W[i] * error) / (1.0 - error)

            # normalize data weights
            self.normalize_data_weights()

            if error == 0:
                self.z[k] = np.log2(float('inf'))
            else:
                self.z[k] = np.log2((1.0 - error) / error)

        # calculate metric for the training set and print
        self.weighted_sum(X_original_train_df, y_original_train_df)

    def predict(self):
        self.x_test, self.y_test = self.convert_df_to_np(self.test_df)
        self.weighted_sum(self.x_test, self.y_test)

    @staticmethod
    def majority_voting(preds):
        axis = 1
        unique_vals, indices = np.unique(preds, return_inverse=True)
        return unique_vals[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(preds.shape),
                                                         None, np.max(indices) + 1), axis=axis)]

    def weighted_sum(self, x, y):

        preds = []
        for k in range(self.num_of_learner):
            weak_learner_weights = self.h[k].weights
            weak_learner_preds = np.tanh(x @ weak_learner_weights)
            weighted_preds = self.z[k] * weak_learner_preds
            preds.append(weighted_preds)

        preds = np.array(preds).squeeze().T
        #print(preds.shape)

        weighted_sum_result = np.sum(preds, axis=1)
        weighted_sum_result[weighted_sum_result > 0] = 1
        weighted_sum_result[weighted_sum_result <= 0] = -1

        metric = MetricCalculator(y, weighted_sum_result)
        metric.calculate_all_metric()

    def normalize_data_weights(self):
        total_data_weight = sum(self.W)
        W = [(data_weight / total_data_weight) for data_weight in self.W]
        self.W = W


def run_churn_adaboost(df):
    print('Adaboost Training Metric (Churn Dataset)')
    adaboost_classifier = Adaboost(num_of_learner=10, test_size=0.2)
    adaboost_classifier.split_dataset(df=df, label='Churn')
    adaboost_classifier.fit(base_learner_max_iter=1000, error=0.5, learning_rate=0.1, decay=5e-6)

    print('Adaboost Testing Metric (Churn Dataset)')
    adaboost_classifier.predict()


def run_churn_lgr(df):
    print('Logistic Regression Training Metric (Churn Dataset)')
    lgr = LogisticRegression(learning_rate=0.1, max_iter=1000, test_size=0.2, early_stop_error=0, decay=5e-6)
    lgr.split_dataset(df, 'Churn')
    lgr.fit(is_constant_lr=False, no_curve=False, calculate_metric_on_train=True)

    print('Logistic Regression Testing Metric (Churn Dataset)')
    lgr.generate_metric()


def run_churn_all(df):
    run_churn_lgr(df)
    run_churn_adaboost(df)


if __name__ == '__main__':
    df = pd.read_csv('./data/cust_churn.csv')
    df = preprocess_churn_data(df=df, label='Churn', num_of_features=8)

    # running on the first dataset
    run_churn_all(df)
