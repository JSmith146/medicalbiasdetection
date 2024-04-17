#required libraries
import datetime 
import pandas as pd
import pprint as pp
import json
import numpy as np
import os
import yaml
from tqdm import tqdm


# Model Development
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, fbeta_score
from sklearn.metrics import (precision_recall_curve, precision_recall_fscore_support, roc_curve, roc_auc_score,
                             f1_score, auc, accuracy_score, confusion_matrix, precision_score,matthews_corrcoef, 
                             recall_score)
from hyperopt import STATUS_OK, hp, fmin, tpe
from imblearn.over_sampling import SMOTE
import functools

# Set envrionmental variables
LOG_PATH = os.getenv('LOG_PATH')
RANDOM_STATE = 312

def timeseries_train_test_split(df,y_column,train_hours,max_hours):
    """
    Creates training, test, and validation datasets based on time series data

    Parameters:
    df (pd.DataFrame): patient dataset
    y_column (string): column name of the response variable
    train_hours (int): number of time periods to use for training data
    max_hours (int): the maximum number of hours to include in the resulting datasets
    test_size (float): the percentage used to split between the validation and test data

    Returns:
    X_train, X_val, X_test
    y_train, y_val, y_test

    """
    # create a temporary dataset
    tmp = df.copy()
    
    # get number of time periods 
    n = len(tmp)
    
    # set response variable column
    tmp.loc[:,'y'] = tmp[y_column]
    
    # separate training data
    X_train = tmp.iloc[:train_hours,:]
    y_train = X_train['y']
    
    # calculate remaining time periods available using the maximum hours as the cap
    remaining_hours = (n - train_hours) if n < max_hours else (max_hours - train_hours)
    
    # determine time periods for validation data
    validation_hours = int(train_hours  + remaining_hours)
    
    # separate validation data
    X_val = tmp.iloc[train_hours:validation_hours,:]
    y_val = X_val['y']
    
    return X_train, y_train, X_val, y_val

def downsample(x,y_col):
    
    pos = x[x[y_col] == 1]
    neg = x[x[y_col] == 0]
    
    if len(pos) < len(neg):
        neg = neg.sample(n=len(pos), replace = False, random_state = RANDOM_STATE)
        
    new = pos.append(neg)
    new = new.sample(frac = 1, replace = False)
    
    return new

def BO_TPE(X_train, y_train, X_val, y_val, loss_function):
    """
    Bayesian Optimization with Tree-structured Parzen Estimator
    
    Parameters:
    X_train (pd.DataFrame)
    y_train (pd.DataFrame)
    X_val (pd.DataFrame)
    y_val (pd.DataFrame)
    """
    
    # Convert datasets to DMatrix format for use with XGBoost
    train = xgb.DMatrix(X_train, label=y_train)
    val = xgb.DMatrix(X_val, label=y_val)
    X_val_D = xgb.DMatrix(X_val)

    def objective(params):
        """
        Objective function to guide the optimization process
        """
        
        # train the model using the provided hyperparameters for a maximum of 1000 iterations with early stopping after 80 iterations without improvement
        xgb_model = xgb.train(params, dtrain=train, num_boost_round=1000, evals=[(val, 'eval')],
                              verbose_eval=False, early_stopping_rounds=80)
        # make predictions using tuned model
        y_vd_pred = xgb_model.predict(X_val_D, ntree_limit=xgb_model.best_ntree_limit)
        # classify responses using 50% threshold
        y_val_class = [0 if i <= 0.5 else 1 for i in y_vd_pred]
        
        # # compute accuracy and loss
        # acc = accuracy_score(y_val, y_val_class)
        # loss = 1 - acc
        
        # # compute F2 score
        # f2 = fbeta_score(y_val, y_val_class, beta=2)
        # calculate loss
        loss = 1 - loss_function(y_val, y_val_class)

        return {'loss': loss, 'params': params, 'status': STATUS_OK}

    # establish bounds for hyperparameter tuning
    max_depths = [3, 4, 5]
    learning_rates = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]
    subsamples = [0.5, 0.6, 0.7, 0.8, 0.9]
    colsample_bytrees = [0.5, 0.6, 0.7, 0.8, 0.9]
    reg_alphas = [0.0, 0.005, 0.01, 0.05, 0.1]
    reg_lambdas = [0.8, 1, 1.5, 2, 4]
    
    # define the search space for hyper parameters
    space = {
        'max_depth': hp.choice('max_depth', max_depths),
        'learning_rate': hp.choice('learning_rate', learning_rates),
        'subsample': hp.choice('subsample', subsamples),
        'colsample_bytree': hp.choice('colsample_bytree', colsample_bytrees),
        'reg_alpha': hp.choice('reg_alpha', reg_alphas),
        'reg_lambda': hp.choice('reg_lambda', reg_lambdas),
    }
    
    # perform optimization
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20)
    
    # define dictionary to house tuned hyperparameters
    best_param = {'max_depth': max_depths[(best['max_depth'])],
                  'learning_rate': learning_rates[(best['learning_rate'])],
                  'subsample': subsamples[(best['subsample'])],
                  'colsample_bytree': colsample_bytrees[(best['colsample_bytree'])],
                  'reg_alpha': reg_alphas[(best['reg_alpha'])],
                  'reg_lambda': reg_lambdas[(best['reg_lambda'])]
                  }

    return best_param

def train_model(k, X_train, y_train, X_val, y_val, loss_function_name, save_model_dir):
    """
    Train the XGBoost algoritm
    """
    print(f"Loss function being used: {loss_function_name}")
    print('*************************************************************')
    print('{}th training ..............'.format(k + 1))
    print('Hyperparameters optimization')
    
    # create loss function map
    loss_functions = {
         'Accuracy':accuracy_score,
         'F1_Score': functools.partial(fbeta_score, beta=1),
         'F2_Score': functools.partial(fbeta_score, beta=2),
         'ROC_AUC_Score': roc_auc_score,
        }
    
    if loss_function_name not in loss_functions:
        raise ValueError(f"Invalid loss function name: {loss_function_name}. Available options are: {list(loss_functions.keys())}")
    
    L = loss_functions[loss_function_name]
    
    # find the best hyperparameters for the model
    best_param = BO_TPE(X_train, y_train, X_val, y_val, L)
    print("obtained best_param")
    
    # initialize XGBoost classification model using the best parameters
    xgb_model = xgb.XGBClassifier(max_depth = best_param['max_depth'],
                                  eta = best_param['learning_rate'],
                                  n_estimators = 1000,
                                  subsample = best_param['subsample'],
                                  colsample_bytree = best_param['colsample_bytree'],
                                  reg_alpha = best_param['reg_alpha'],
                                  reg_lambda = best_param['reg_lambda'],
                                  objective = "binary:logistic",
                                  use_label_encoder=False
                                  )
    
    # fit the XGBoost classification model
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='error',
                  early_stopping_rounds=80, verbose=False)
    # compute predictions
    y_tr_pred = (xgb_model.predict_proba(X_train, ntree_limit=xgb_model.best_ntree_limit))[:, 1]
    
    # Evaluate model on training data
    train_auc = roc_auc_score(y_train, y_tr_pred)
    print('training dataset AUC: ' + str(train_auc))
    
    # convert probabilites to binary responses
    y_tr_class = [0 if i <= 0.5 else 1 for i in y_tr_pred]
    acc = accuracy_score(y_train, y_tr_class)
    print('training dataset acc: ' + str(acc))
    f2_score = fbeta_score(y_train, y_tr_class,beta=2.0)
    print('training dataset f2-score: ' + str(f2_score))
    
    # Evaluate model on validation data
    y_vd_pred = (xgb_model.predict_proba(X_val, ntree_limit=xgb_model.best_ntree_limit))[:, 1]
    
    # calculate the AUC score
    valid_auc = roc_auc_score(y_val, y_vd_pred)
    print('validation dataset AUC: ' + str(valid_auc))
    
    # calculate the Accuracy
    y_val_class = [0 if i <= 0.5 else 1 for i in y_vd_pred]
    acc = accuracy_score(y_val, y_val_class)
    print('validation dataset acc: ' + str(acc))
    f2_score = fbeta_score(y_val, y_val_class,beta=2.0)
    print('validation dataset f2-score: ' + str(f2_score))
    print('************************************************************')
    
    # save the model
    save_path = os.path.join(save_model_dir,f"y_train_{str(k)}.npy")
    np.save(save_path, y_train)
    
    save_path = os.path.join(save_model_dir,f"y_train_pred_{str(k)}.npy")
    np.save(save_path, y_tr_pred)
    
    save_path = os.path.join(save_model_dir,f"y_val_{str(k)}.npy")
    np.save(save_path, y_val)
    
    save_path = os.path.join(save_model_dir,f"y_val_pred_{str(k)}.npy")
    np.save(save_path, y_vd_pred)
    
    save_model_path = os.path.join(save_model_dir, f"model_{k + 1}.mdl")
    xgb_model.get_booster().save_model(fname=save_model_path)
    
    
def load_model_predict(X_test, k_fold, path):
    "ensemble the five XGBoost models by averaging their output probabilities"
    test_pred = np.zeros((X_test.shape[0], k_fold))
    X_test = xgb.DMatrix(X_test)
    for k in range(k_fold):
        model_path_name = os.path.join(path,f'model_{k+1}.mdl')
        xgb_model = xgb.Booster(model_file = model_path_name)
        y_test_pred = xgb_model.predict(X_test)
        test_pred[:, k] = y_test_pred
    test_pred = pd.DataFrame(test_pred)
    result_pro = test_pred.mean(axis=1)

    return result_pro


def predict(patient_csns, X, model_path, k_folds, outpath, risk_threshold, drop_cols):
    
    # initialize results dataframe
    result = pd.DataFrame()

    # setup progress bar
    pbar = tqdm(total = len(patient_csns), position=0, leave=True)
    
    # for each patient in the test set
    for csn in patient_csns:
        try:
            # filter patient by csn
            pat_df = X[X["csn"] == csn]

            # continue if dataframe is empty
            if pat_df.shape[0] ==0:
                continue

            # get labels
            labels = pat_df['y']

            # drop columns
            features = pat_df.drop(columns=drop_cols, errors='ignore').values

            # predict labels using test data
            y_pred = load_model_predict(features, k_fold = k_folds, path = model_path)

            # convert predictions to numpy array
            PredictedProbability = np.array(y_pred)

            # convert prediction proabbilites to labels
            PredictedLabel = [0 if i <= risk_threshold else 1 for i in y_pred]

            tmp_df = pat_df.copy()

            tmp_df = tmp_df[['csn']]

            tmp_df["y_prob"] = PredictedProbability

            tmp_df["y_pred"] = PredictedLabel

            tmp_df["y_true"] = labels

            tmp_df.to_csv(outpath, mode='a', header=not os.path.exists(outpath))

            result = result.append(tmp_df)
        except:
            continue
        # update progress bar
        pbar.update(1)
    
    #close progress bar
    pbar.close()
    
    return result

def calculate_confusion_matrix(y_test, y_pred):
    """
    Calculate the confusion matrix for a given y_test and y_pred.

    Args:
    y_test: The true labels.
    y_pred: The predicted labels.

    Returns:
    A confusion matrix.
    """
    confusion_matrix = np.zeros((2, 2))
    for i in range(len(y_test)):
        if y_test[i] == 0 and y_pred[i] == 0:
            confusion_matrix[0, 0] += 1
        elif y_test[i] == 0 and y_pred[i] == 1: #false positive
            confusion_matrix[0, 1] += 1
        elif y_test[i] == 1 and y_pred[i] == 0: #false negative
            confusion_matrix[1, 0] += 1
        elif y_test[i] == 1 and y_pred[i] == 1:
            confusion_matrix[1, 1] += 1
    return confusion_matrix

def summarize_predictions(df,y_test_col,y_pred_col,risk_threshold):

    csns = df["csn"].unique().tolist()
    results = pd.DataFrame()

    for csn in csns:
        metrics = {}
        metrics["csn"] = str(csn)
        tmp = df.loc[df["csn"]==csn,[y_test_col,y_pred_col]].reset_index()
        tmp["y_test"] = tmp[y_test_col].astype('int')
        tmp["y_pred"] = [0 if i <= risk_threshold else 1 for i in tmp[y_pred_col]]
        fpr, tpr, thresholds = roc_curve(tmp["y_test"], tmp["y_pred"])
        
        accuracy = accuracy_score(tmp["y_test"], tmp["y_pred"])
        f2_score = fbeta_score(tmp['y_test'],tmp['y_pred'],beta=2.0)
        mcc = matthews_corrcoef(tmp['y_test'],tmp['y_pred'])
        cm = calculate_confusion_matrix(tmp["y_test"], tmp["y_pred"])
        metrics["true_negative"] = cm[0][0] #true_negative
        metrics["false_positive"] = cm[0][1] #false_positive
        metrics["false_negative"] = cm[1][0] #false_negative
        metrics["true_positive"] = cm[1][1] #true_positive
        metrics["tpr"] = tpr[1]
        metrics["fpr"] = fpr[1]
        metrics["mcc"] = mcc
        metrics["f1_score"] = f1_score(tmp["y_test"], tmp["y_pred"])
        metrics["f2_score"] = f2_score
        metrics["precision"] = precision_score(tmp["y_test"], tmp["y_pred"])
        metrics["recall"] = recall_score(tmp["y_test"], tmp["y_pred"])
        metrics["accuracy"] = accuracy
        metrics['roc_auc'] = np.nan
        try:
            metrics['roc_auc'] = roc_auc_score(tmp["y_test"], tmp["y_pred"])
        except ValueError:
            pass
            
        metrics = pd.DataFrame.from_dict(metrics,orient='index').T
        results = pd.concat([results,metrics],0,ignore_index=True)
    return results


def load_model_predict_i(X_test, k_fold, path):
    """Get the output probability of a single XGBoost model"""
    test_pred = np.zeros((X_test.shape[0], k_fold))
    X_test = xgb.DMatrix(X_test)
    for k in range(k_fold):
        # get model path
        model_path_name = os.path.join(path,f'model_{k+1}.mdl')
        # get model
        xgb_model = xgb.Booster(model_file = model_path_name)
        # make predictions
        y_test_pred = xgb_model.predict(X_test)
        
        test_pred[:, k] = y_test_pred
    test_pred = pd.DataFrame(test_pred)
    # result_pro = test_pred.mean(axis=1)

    # return result_pro
    return test_pred


def predict_i(patient_csns, X, model_path, k_folds, outpath, risk_threshold, drop_cols):
    """Make predictions for a single XGBoost model"""
    # initialize results dataframe
    results = []

    # setup progress bar
    pbar = tqdm(total = len(patient_csns), position=0, leave=True)
    n = 0
    
    # for each patient in the test set
    for csn in patient_csns:
        try:
            # filter patient by csn
            pat_df = X[X["csn"] == csn]

            # continue if dataframe is empty
            if pat_df.shape[0] ==0:
                continue

            # get labels
            labels = pat_df['y']

            # drop columns
            features = pat_df.drop(columns=drop_cols, errors='ignore').values

            # predict labels using test data
            y_pred = load_model_predict_i(features, k_fold = k_folds, path = model_path)

            # convert predictions to numpy array
            PredictedProbability = np.array(y_pred)

            # convert prediction proabbilites to labels
            PredictedLabel = np.where(PredictedProbability < risk_threshold, 0, 1)

            # create column names for predicted probabilities and lables
            PredictedLabelNames = [f"label_{i+1}" for i in range(k_folds)]
            PredictedProbNames = [f"prob_{i+1}" for i in range(k_folds)]

            # create dataframe for predicted Probabilites
            pred_prob_df = pd.DataFrame(PredictedProbability, columns=PredictedProbNames)

            # create dataframe for predicted labels
            pred_label_df = pd.DataFrame(PredictedLabel,columns=PredictedLabelNames)

            # merged predicted probabilites and labels
            pred_merged_df = pd.concat([pred_prob_df,pred_label_df],axis=1)

            # add true labels and csns to merged data
            pred_merged_df['csn'] = csn
            pred_merged_df['y_true'] = labels.values
            
            pred_merged_df.to_csv(outpath, mode='a', header=not os.path.exists(outpath))
            
            results.append(pred_merged_df)
            
        except:
            continue
        # update progress bar
        pbar.update(1)
    
    #close progress bar
    pbar.close()
    
    result_df = pd.concat(results, axis=0)
    
    return result_df

def summarize_predictions_i(df,y_test_col,y_pred_col,risk_threshold):
    """Calculate prediction results for a single model"""
    
    # init results dataframe
    results = pd.DataFrame()
    # init dictionary to store metric results
    metrics = {}
    # init dataframe to store predictions
    tmp = pd.DataFrame()
    tmp["y_test"] = df[y_test_col].astype('int')
    tmp["y_pred"] = np.where(df[y_pred_col]<risk_threshold,0,1)
    
    # calculate performance metrics
    fpr, tpr, thresholds = roc_curve(tmp["y_test"], tmp["y_pred"])
    mcc = matthews_corrcoef(tmp['y_test'],tmp['y_pred'])
    accuracy = accuracy_score(tmp["y_test"], tmp["y_pred"])
    f2_score = fbeta_score(tmp['y_test'],tmp['y_pred'],beta=2.0)
    cm = calculate_confusion_matrix(tmp["y_test"].values, tmp["y_pred"].values)
    
    # store performance metrics
    metrics["true_negative"] = cm[0][0] #true_negative
    metrics["false_positive"] = cm[0][1] #false_positive
    metrics["false_negative"] = cm[1][0] #false_negative
    metrics["true_positive"] = cm[1][1] #true_positive
    metrics["mcc"] = mcc
    metrics["tpr"] = tpr[1]
    metrics["fpr"] = fpr[1]
    metrics["f1_score"] = f1_score(tmp["y_test"], tmp["y_pred"])
    metrics["f2_score"] = f2_score
    metrics["precision"] = precision_score(tmp["y_test"], tmp["y_pred"])
    metrics["recall"] = recall_score(tmp["y_test"], tmp["y_pred"])
    metrics["accuracy"] = accuracy
    metrics['roc_auc'] = np.nan
    try:
        metrics['roc_auc'] = roc_auc_score(tmp["y_test"], tmp["y_pred"])
    except ValueError:
        pass

    results = pd.DataFrame.from_dict(metrics,orient='index').T
    
    return results