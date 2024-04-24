#required libraries
import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

# Set envrionmental variables
LOG_PATH = os.getenv('LOG_PATH')
RUN = int(os.getenv('RUN'))

def load_reference_data(med_fac, config, verbose=False):

    # get columns that will be converted to datetime 
    time_cols = config['meta']['time_cols']

    # get associated data types for each column
    with open(config['data']['sep3_dtypes'],"r") as file:
        data_types = json.load(file)

    # get lab and treatment labels    
    with open(config['data']['lab_treatment_dict'],"r") as file:
        lab_trt_dict = json.load(file)

    ## Elixhauser Comorbidity Data
    CMBI = config['meta']['elixhauser']

    # Static patient data
    path  = config['data'][med_fac]['static']
    data = pd.read_csv( path, parse_dates=True, index_col=0)

    for col in  data.columns[data.columns.isin(time_cols)]:
        data[col] = pd.to_datetime(data[col])

    X = data.copy()
    
    if verbose:
        print_cohort_report(X,'csn','pat_id','sepsis')  
    return X

def load_hourly_data(config):
    # get directory
    data_dir = config['DIR']['data'].format(RUN=RUN,TYPE='hourly')

    # list of patient data files
    files = [f for f in os.listdir(data_dir) if "processed_data_" in f]

    data_arr = []
    for file in tqdm(files, leave=True):
        path = os.path.join(data_dir,file)
        tmp = pd.read_csv(path,index_col=[0])
        data_arr.append(tmp)

    # create dataframe
    df = pd.concat(data_arr,axis=0)
    
    return df


def print_cohort_report(df,encounter_id,patient_id,category):

    n_pat_visits = df.loc[:,encounter_id].unique().shape[0]
    n_pat_id = df.loc[:,patient_id].unique().shape[0]
    cat_labels = df[category].unique().tolist()
    years = df['start_index'].apply(lambda x:x.year).unique()
    print(f"Number of encounters (csn): {df.shape[0]}")
    print(f"Years: {years}")
    print(f"Start year: {years.min()}")
    print(f"End year: {years.max()}")
    print(f"Number of unique patient visits: {n_pat_visits}")
    print(f"Number of unique patients: {n_pat_id}")
    [print(f"Number of {category}={x} patients: {df[df[category]==x].shape[0]} ({100*(df[df[category]==x].shape[0]/df.shape[0]):.2f}%)")  for x in cat_labels]
    pass



def update_cohort(dataframe,log_file_path=LOG_PATH, verbose=False):
    
    # load log data
    log_df = pd.read_csv(log_file_path, index_col=0)

    # get list of csns to include
    include_csns = log_df[log_df['include']==True]['csn'].tolist()

    # print excluded csns history
    excluded_csns = log_df[log_df['include']==False][['step','reason']]

    if verbose:
        print(f"Total CSNs included: {len(include_csns)}")
        print("Removed CSNs:")
        print(excluded_csns[['step','reason']].value_counts())

    # remove csns from data
    result = dataframe[dataframe['csn'].isin(include_csns)].copy()

    return result


def create_csn_log(dataframe, outpath):
    """
    Creates initial log for all patient csns. This file is meant to be updated whenever patients are removed
    from the cohort. The step at which the patient visit is removed and removal reasoning are to be logged also.
    
    Parameters:
    dataframe (pd.DataFrame): The starting data comprising the entire cohort
    outpath (str): The file location to save the resulting csv file
    
    Returns:
    None
    """
    # check if the outpath file location exists, if not create it
    if not os.path.exists(outpath):
        os.makedirs(outpath,exist_ok=True)
        print("Log file path created")
    
    
    # create a copy of the dataframe to prevent changing original data
    X = dataframe.copy()
    
    # create a csn dataframe
    csn_df = pd.DataFrame(X['csn'].unique().tolist()).rename(columns={0:'csn'})
    
    # add additional column information
    csn_df['include'] = True # bool to determine if csn is currently included
    csn_df['step'] = ''      # identifies when the csn was removed
    csn_df['reason'] = ''    # identifies why the csn was removed
    csn_df['n'] = 0          # used to check that a cns was not removed more than once
    
    # filename to save data
    filename = "csn_log.csv"
    save_path = os.path.join(outpath,filename)
    
    # write csn log file to directory
    csn_df.to_csv(save_path,header=True,index=True)
    print(f"Log file created at: \n {save_path}")
    
    

def log_removals(csn_to_remove, step, reason, log_file=LOG_PATH):
        
    # Read the log file into a DataFrame
    csn_df = pd.read_csv(log_file, index_col=0)
    
    # Determine which csns to update based on 'csn_to_remove'
    mask = csn_df['csn'].isin(csn_to_remove)
    # Proceed only if there are matches
    if mask.any():  
        # update all relevant columns for the IDs to be removed
        csn_df.loc[mask, ['include', 'step', 'reason', 'n']] = [False, step, reason, csn_df.loc[mask, 'n'] + 1]
        # Ssave the updates back to the CSV log file
        csn_df.to_csv(log_file, header=True, index=True)
        print("Log file updated.")

def csn_removal_logger(func):
        def wrapper(dataframe, csn_to_remove, step, reason, *args, **kwargs):
            # Log the removal of these IDs if not empty
            log_removals(csn_to_remove, step, reason)
            # Remove the IDs from the DataFrame and return the updated DataFrame
            result = func(dataframe, csn_to_remove, *args, **kwargs)
            return result
        return wrapper

@csn_removal_logger
def remove_csn(dataframe, csn_to_remove, *args, **kwargs):
    # Filter out the IDs from the DataFrame
    return dataframe[~dataframe['csn'].isin(csn_to_remove)]


def load_cohort(med_fac="grady", log_file_path=LOG_PATH, verbose=False):
    # identify the medical facility for the dataset
    med_fac = 'grady'

    # get columns that will be converted to datetime 
    time_cols = config['meta']['time_cols']

    # get associated data types for each column
    with open(config['data']['sep3_dtypes'],"r") as file:
        data_types = json.load(file)

    # get lab and treatment labels    
    with open(config['data']['lab_treatment_dict'],"r") as file:
        lab_trt_dict = json.load(file)

    ## Elixhauser Comorbidity Data
    CMBI = config['meta']['elixhauser']

    data = pd.read_csv('data/cohortdata/grady/grady_static.csv', parse_dates=True, index_col=0)

    for col in  data.columns[data.columns.isin(time_cols)]:
        data[col] = pd.to_datetime(data[col])

    # load log data
    log_df = pd.read_csv(LOG_PATH, index_col=0)

    # get list of csns to include
    include_csns = log_df[log_df['include']==True]['csn'].tolist()

    # print excluded csns history
    excluded_csns = log_df[log_df['include']==False][['step','reason']]

    if verbose:
        print("Removed CSNs:")
        print(excluded_csns[['step','reason']].value_counts())

    # remove csns from data
    result = data[data['csn'].isin(include_csns)]

    return result


def right_censor(dataframe, sepsis_col):
    """
    Right-censor hourly sepsis data of each patient
    Params: 
        - df: hourly sepsis data
        - sepsis_col (str): observed sepsis label for the given patient hour
    """
    
    df = dataframe.copy()
    
    # identify each unique csn
    csns = df['csn'].unique()
    
    # initiate list to hold censored results 
    rc_df_arr = []
    
    # iterate through each patient visit and censor predictions
    for csn in csns:
        
        tmp = df[df['csn']==csn].copy()
        
        # create flag for positive sepsis hours
        sepsis = tmp[sepsis_col].sum() > 0.0

        if sepsis:
            # identify index of first positive sepsis hour
            idx = tmp[tmp[sepsis_col]==1].iloc[0].name
            
            # adjust csn dataframe to stop at the first sepsis hour
            tmp = tmp.loc[:idx]
            
        # append resulting dataframe to results
        # limit possible hours up to 168
        rc_df_arr.append(tmp.iloc[:168])
        
    # concatenate right-censored predictions    
    rc_df = pd.concat(rc_df_arr, axis=0)
    
    return rc_df

def oh_encode_pat_data(df, dummy_cols):
    # copy df
    clean_df = df.copy()
    # loop through each dummy column
    for col in dummy_cols:
        if col == 'admit_reason':
            dummy = pd.get_dummies(clean_df[col],prefix='AR')
        elif col == 'gender':
            # clean gender column
            clean_df[col] = clean_df[col].replace({0:'male', 1:'female'})
            # create dummy variables
            dummy = pd.get_dummies(clean_df[col])
        else:
            dummy = pd.get_dummies(clean_df[col])
        # remove originating column
        clean_df.drop(columns=col,inplace=True)
        # add dummy columns to clean data
        clean_df[dummy.columns] = dummy
    
    return clean_df