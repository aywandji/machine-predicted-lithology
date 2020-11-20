from joblib import dump, load
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def reduce_mem_usage(props):
    """
    Reduce pandas DataFrame size.
    """
    start_mem_usg = props.memory_usage().sum() / 1024**2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in props.columns:

        if props[col].dtype != object:  # Exclude strings

            # Print current column type
            print("******************************")
            print("Column: ", col)
            print("dtype before: ", props[col].dtype)

            # make variables for Int, max and min
            IsInt = "int" in str(props[col].dtype)
            mx = props[col].max()
            mn = props[col].min()

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

            # Print new column type
            print("dtype after: ", props[col].dtype)
            print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return props, NAlist


def penalty_score(y_true, y_pred,penalty_matrix=None):
    S = 0.0
    lithology_cat_to_position = {30000: 0, 65030: 1, 65000: 2, 80000: 3, 74000: 4, 70000: 5,
                                 70032: 6, 88000: 7, 86000: 8, 99000: 9, 90000: 10, 93000: 11}
    for i in range(y_true.shape[0]):
        r = lithology_cat_to_position[y_true[i]]
        c = lithology_cat_to_position[y_pred[i]]
        S -= penalty_matrix[r, c]
    return S / y_true.shape[0]


def plot_features_importance(f_importances, features_names, figsize=(20, 20)):
    features_df = pd.Series(data=f_importances, index=features_names)
    features_df = features_df.sort_values(ascending=False)

    plt.figure(figsize=figsize)
    sns.barplot(x=features_df, y=features_df.index)

def predict_target(target_val_x, scaler, num_features,reg):
    """
    Scale features and predict target.
    """
    target_val_x.loc[:, num_features] = scaler.transform(target_val_x[num_features])
    val_pred = reg.predict(target_val_x)

    return val_pred, target_val_x

def get_predictor(processed_df,catboost_classifier,target='', regressors=[],encoded_cat=[]):
    """
        Fill "target" column by training a "catboost_classifier" using "regressors" as features.
    """
    mask_to_fill = processed_df[target].isna()
    for regr in regressors:
        mask_to_fill = mask_to_fill & processed_df[regr].notna()

    if np.sum(mask_to_fill) == 0:
        print("no sample with all regressors {} available to predict empty target {}. Or no empty target".format(
            regressors, target))
        print("----------------\n")

        return processed_df, None

    # TRAINING DATA (where target and regressors are not null. We use only non null original target (from train_df))
    predictors = regressors + [target]
    original_non_null_target_mask = processed_df.loc[:, target].notna()
    non_null_target_samples = processed_df.loc[original_non_null_target_mask, predictors]
    
    mask = True
    for pred in regressors:
        mask = mask & non_null_target_samples[pred].notna()

    target_data = non_null_target_samples[mask]
    target_train, target_val = train_test_split(target_data, test_size=0.2)
    
    # DATA SCALING
    target_x = target_train.drop(columns=target)
    num_features = target_x.columns.drop(encoded_cat)
    
    scaler = RobustScaler()
    target_x.loc[:, num_features] = scaler.fit_transform(target_x[num_features])

    target_y = target_train.loc[:, target]
    
     # VALIDATION DATA
    target_val_y = target_val.loc[:, target]
    target_val_x = target_val.drop(columns=target)
    
    target_val_x.loc[:, num_features] = scaler.transform(target_val_x[num_features])

    catboost_classifier.fit(
          target_x, target_y,
          eval_set=[(target_val_x, target_val_y)],
          verbose=10)


    results = catboost_classifier.get_best_score()
    print(results["learn"]["Accuracy"])    
    print(results["validation"]["Accuracy"])

    empty_target = processed_df.loc[mask_to_fill, regressors]

    predicted_target, _ = predict_target(
        empty_target, scaler,num_features, catboost_classifier)

    processed_df.loc[mask_to_fill, target] = predicted_target
    mask_to_fill = processed_df[target].isna()
    print("there are still {} values to fill".format(np.sum(mask_to_fill)))
    print("----------------\n")

    return processed_df, catboost_classifier, scaler
class WellLogsProcessing():

    def __init__(self, numeric_features=[], cat_features=[], others_numeric=[], all_formations=[], all_groups=[],
                 remove_outliers=False, process_num_features=False,
                 impute_categorical=False, encode_categorical=False):

        self.numeric_features = list(numeric_features)  # to impute and scale
        # to impute and onehot (if specified)
        self.cat_features = list(cat_features)
        # to  scale and add to final processed df
        self.others_numeric = list(others_numeric)
        self.all_features = self.numeric_features + \
            self.others_numeric + self.cat_features + ["WELL"]

        self.outliers_thresholds = {}
        self.iterative_imputer = None
        self.data_scaler = None

        self.cat_transform_method = "ordinal_encoding"
        self.ordinal_encoder = None
        self.onehot_encoder = None

        # self.all_formations = defaultdict(lambda:100)
        # self.all_formations = defaultdict(lambda:100)
        # self.all_groups = defaultdict(lambda:100)
        self.all_formations = {v: k for k,
                               v in enumerate(sorted(all_formations))}
        self.all_groups = {v: k for k, v in enumerate(sorted(all_groups))}

        self.remove_outliers = remove_outliers
        self.process_num_features = process_num_features
        self.impute_categorical = impute_categorical
        self.encode_categorical = encode_categorical

    def load_processing_params(self, processing_params_path):
        processing_params = np.load(
            processing_params_path, allow_pickle=True).item()

        self.outliers_thresholds = processing_params['outliers_thresh']
        self.data_scaler = load(processing_params['scaler_path'])
        self.iterative_imputer = load(processing_params['imputer_path'])

        cat_encoder = load(processing_params['cat_encoder_path'])

        self.cat_transform_method = processing_params['cat_transform_method']
        if self.cat_transform_method == "ordinal_encoding":
            # self.ordinal_encoder = cat_encoder
            self.all_formations = cat_encoder["FORMATION"]
            self.all_groups = cat_encoder["GROUP"]
        else:
            self.onehot_encoder = cat_encoder

        # to impute and scale
        self.numeric_features = processing_params['numeric_features']
        # to impute and onehot (if specified)
        self.cat_features = processing_params['cat_features']
        # to  scale and add to final processed df
        self.others_numeric = processing_params['others_numeric']

        self.all_features = self.numeric_features + \
            self.others_numeric + self.cat_features + ["WELL"]

    def save_processing_params(self, data_folder="force_ml_data/", scaler_name="data_scaler", imputer_name="iterative_imputer", cat_encoder_name="cat_encoder", params_name="processing_params"):
        scaler_path = data_folder + scaler_name + '.joblib'
        imputer_path = data_folder + imputer_name + '.joblib'
        cat_encoder_path = data_folder + cat_encoder_name + '.joblib'

        dump(self.data_scaler, scaler_path)
        dump(self.iterative_imputer, imputer_path)

        # if self.onehot_encoder is None:
        if self.cat_transform_method == "ordinal_encoder":
            encoder_dict = {}
            encoder_dict["FORMATION"] = self.all_formations
            encoder_dict["GROUP"] = self.all_groups

            dump(encoder_dict, cat_encoder_path)
        else:
            dump(self.onehot_encoder, cat_encoder_path)

        processing_params = {}
        processing_params['scaler_path'] = scaler_path
        processing_params['imputer_path'] = imputer_path
        processing_params['cat_encoder_path'] = cat_encoder_path
        processing_params['cat_transform_method'] = self.cat_transform_method
        processing_params['outliers_thresh'] = self.outliers_thresholds
        processing_params['numeric_features'] = self.numeric_features
        processing_params['cat_features'] = self.cat_features
        processing_params['others_numeric'] = self.others_numeric
        np.save(data_folder + params_name + '.npy', processing_params)

    def get_processed_data(self, logs_df, is_train_data=False, method="ordinal_encoding", cat_imputer=None):
        if self.remove_outliers:
            logs_df = self.remove_outliers(logs_df, is_train_data)

        if self.process_num_features:
            logs_df = self.process_numerical_features(logs_df, is_train_data)

        logs_df = self.process_categorical_features(
            logs_df, is_train_data, method=method, cat_imputer=cat_imputer)

        return logs_df[self.all_features]

    def remove_outliers(self, logs_df, is_train_data):
        # threshold values to remove outliers
        for col in self.numeric_features:
            df = logs_df.loc[:, col]

            if is_train_data:
                print("generating outliers threshold for feature {}".format(col))

                Q1 = df.quantile(0.25)
                Q3 = df.quantile(0.75)
                IQR = Q3 - Q1
                tmin = Q1 - 1.5 * IQR
                tmax = Q3 + 1.5 * IQR
                self.outliers_thresholds[col] = [tmin, tmax]

            tmin, tmax = self.outliers_thresholds[col]

            logs_df.loc[df > tmax, col] = tmax
            logs_df.loc[df < tmin, col] = tmin

        return logs_df

    def scale_data(self, logs_df, is_train_data):
        full_numeric = self.numeric_features + self.others_numeric
        if is_train_data:
            print("Fitting data scaler")

            self.data_scaler = RobustScaler()
            self.data_scaler.fit(logs_df[full_numeric].values)

        logs_df.loc[:, full_numeric] = self.data_scaler.transform(
            logs_df.loc[:, full_numeric])

        return logs_df

    # imputation  + scaling
    def process_numerical_features(self, logs_df, is_train_data):
        """
        Fit iterative imputer to fill empty values. Then scale data.
        """
        if is_train_data:
            print("Fitting iterative imputer")

            self.iterative_imputer = IterativeImputer(n_nearest_features=5,
                                                      initial_strategy='mean',
                                                      imputation_order='ascending',
                                                      verbose=1,
                                                      random_state=0)

            self.iterative_imputer.fit(logs_df[self.numeric_features].values)

        logs_df.loc[:, self.numeric_features] = self.iterative_imputer.transform(
            logs_df[self.numeric_features].values)

        logs_df = self.scale_data(logs_df, is_train_data)

        return logs_df

    def transform_categorical_features(self, logs_df, is_train_data, method="ordinal_encoding"):
        if is_train_data:
            print("Setting categorical encoding method... to {}".format(method))

            self.cat_transform_method = method

        if self.cat_transform_method == "ordinal_encoding":
            self.onehot_encoder = None
            for cat_f in self.cat_features:
                if cat_f == "FORMATION":
                    dict_map = self.all_formations
                else:
                    dict_map = self.all_groups

                logs_df.loc[:, cat_f] = logs_df.loc[:, cat_f].map(dict_map)
                # map returns Nan when value not in dict
                na_mask = logs_df[cat_f].isna()
                # value for categorical not found in training values
                logs_df.loc[na_mask, cat_f] = 1000

        else:
            self.ordinal_encoder = None
            if is_train_data:
                print("Fitting onehot encoder...")
                self.onehot_encoder = OneHotEncoder()
                self.onehot_encoder.fit(logs_df.loc[:, self.cat_features])

            onehot_columns = self.onehot_encoder.get_feature_names(
                self.cat_features)
            logs_df.loc[:, onehot_columns] = self.onehot_encoder.transform(
                logs_df.loc[:, self.cat_features])

            # drop cat_features
            logs_df.drop(columns=self.cat_features, inplace=True)

        return logs_df

    # imputation  + encoding
    def process_categorical_features(self, logs_df, is_train_data, method, cat_imputer):
        """
        Impute categorical features with mode and/or encode following encoding "method"
        """
        if self.impute_categorical:
            # impute categorical features
            for well_name in logs_df.WELL.unique():
                mask = logs_df.WELL == well_name
                for cat_feature in self.cat_features:
                    wg = logs_df.loc[mask, cat_feature].mode()[0]
                    logs_df.loc[mask, cat_feature] = logs_df.loc[mask,
                                                                 cat_feature].fillna(wg)

        if self.encode_categorical:
            logs_df = self.transform_categorical_features(
                logs_df, is_train_data, method=method)

        return logs_df
