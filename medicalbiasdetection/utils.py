#required libraries
import os
import yaml

import pandas as pd
import datatable as dt
import numpy as np

import json
import pprint as pp
import datetime




# Set envrionmental variables
LOG_PATH = os.getenv('LOG_PATH')

# Get the year from the date string
def get_year(date):
    return datetime.datetime .strptime(date, '%Y-%m-%d %H:%M:%S').year

def mem_usage(df,verbose=False):
    if isinstance(df,pd.DataFrame):
        usage_b = df.memory_usage(deep=True).sum()
    else:
        usage_b = df.memory_usage(deep=True)
    usage_mb = usage_b/1024**2 # converts to MB
    if verbose:
        print("{:03.2f} MB".format(usage_mb))
    return usage_mb

def find_dtypes(df):
    cols_dtypes = [str(x) for x in df.dtypes.unique()]
    res = set()
    for i in cols_dtypes:
        if "int" in i:
            res.add("int")
        if "float" in i:
            res.add("float")
        if "object" in i:
            res.add("object")
    return res

def find_time_cols(df):
    return [x for x in df.columns if "time" in x]

def time_to_int(time_col):
    time_col = time_col.apply(lambda x: x.split('.')[0])
    converted_time = time_col.apply(lambda x: int(time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S'))))
    return converted_time

def convert_obj_cols(col):
    num_unique_values = len(col.unique())
    num_total_values = len(col)
    if num_unique_values / num_total_values < 0.5:
        conv_obj_col= col.astype('category')
    else:
        conv_obj_col= col
    return conv_obj_col

def optimize_memory(df,verbose=False):
        report = {}
        opt_df = df.copy()
        time_cols = find_time_cols(opt_df)
        opt_df.drop(columns=time_cols, inplace=True)
        
        for col in opt_df.columns:
            if verbose:
                print(f"{col}............")
            col_type = str(opt_df[col].dtype)
            print(col,'------',col_type)
            mem_before = mem_usage(opt_df[col])
            if "object" in col_type:
                opt_df[col] = convert_obj_cols(opt_df[col])
            if "int" in col_type:
                opt_df[col] = opt_df[col].apply(pd.to_numeric,downcast='unsigned')
            if "float" in col_type:
                opt_df[col] = opt_df[col].apply(pd.to_numeric, downcast='float')
                
            mem_after = mem_usage(opt_df[col])
            report[col]={'col_type':col_type,'%reduction':(mem_before-mem_after)/mem_before,'mem_use_before':mem_before,'mem_use_after':mem_after}
           
        if verbose:
            pprint.pprint(report)
        dtypes = opt_df.dtypes
        dtypes_col = dtypes.index
        dtypes_type = [i.name for i in dtypes.values]
        transform_map = dict(zip(dtypes_col,dtypes_type))
        opt_df = pd.concat([opt_df,df[time_cols]],axis=1) 
        
        return opt_df,transform_map, time_cols
    
def print_columns(df):
    [print(f"Name: {x} | Type:{df[x].dtype}") for x in df.columns]
    
    
def convert_timedelta(t_delta):
    seconds = t_delta.total_seconds()
    hours = seconds // 3600
    minutes = seconds / 60
    return t_delta.days, hours, minutes

def count_labs(lab_col):
    col = pd.DataFrame(lab_col)
    col_name = col.columns[0]
    col["diff"]= col[col_name].diff()
    col['counts'] = [1 if (x!=0) else 0 for x in col["diff"]]
    col["isna"] = [-1 if x==True else 0 for x in col[col_name].isna()]
    col['lab_count'] = col['counts']+col["isna"]
    col['lab_counts'] = col['lab_count'].cumsum()
    return col['lab_counts']

def count_totals(df,cols):
    count_cols = [f"{x}_counts" for x in cols]
    total = df[count_cols].sum(axis=1)
    return total
import pandas as pd

def read_csv(file_path, datetime_columns):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Filter out datetime columns that are not present in the dataset
        datetime_columns = [col for col in datetime_columns if col in df.columns]
        
        # Convert datetime columns to datetime data type
        for col in datetime_columns:
            df[col] = pd.to_datetime(df[col])
        
        return df
    
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' could not be found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {str(e)}")
        return None

def get_latest_version(filepath, verbose=False):
    # Create empty list for csv files
    csv_list = []
    for file in os.listdir(filepath):
        if file.endswith("csv"):
            name = file.split('_')[0]
            i = int(file.split("_")[1].split(".")[0])
            csv_list.append(i)
            if verbose:
                print(f"{name}: Version {i}")
    return np.max(i)
            

    
def reduce_memory_usage(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df

def read_yaml(filepath=None):
    """ 
    A function to read YAML file
    Params
    filepath (str): filepath to configuration file destination
    
    Returns
    config (dictionary): the specified configuration file
    """
    if not filepath:
        filepath = 'conf/config.yaml'
    try:
        with open(filepath) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError as e:
        print(f"File {e.filename} not found!")
 
    return config



def create_run_dir(run , root=None, verbose=False):
    """
    Creates real data experiment directory structure based the user specified run identifier
    
    Parameters:
    run (int): The unique ID for the experiment run
    
    Returns:
    None
    """
    def create_subfolders(base, subfolders, verbose = verbose):
        if isinstance(subfolders, list):
            for sub in subfolders:
                sub_path = os.path.join(base, sub)
                if verbose:
                    print(sub_path)
                if not os.path.exists(sub_path):
                    os.makedirs(sub_path)
        elif isinstance(subfolders, str):
            create_subfolders(base, subfolder_dict[subfolders])
            
    if not run:
        # define outpath for directory
        raise ValueError("Must specify a Run ID!")
        
    # ensure run is a string
    run = str(run)
    
    # set the directory root
    if not root:
        root = 'MBD_Runs'
        
    # create filepath
    folder = os.path.join(root,run)
    print(folder)
    
    # subfolder structure
    subfolder_dict = {
        'data':['hourly','skipped','static','log','cases','sep_patient_missingness','tableOne'],
        'models/XGB':'model_types',
        'model_types':['Accuracy','F1_Score','F2_Score'],
        'predictions':'model_types',
        'train_test_split':None,
        'images':'model_types',
        'image_types':['Original','Specificity'],
        'images/Accuracy':'image_types',
        'images/F1_Score':'image_types',
        'images/F2_Score':'image_types',
        'thresholds':['0.0', '0.25', '0.50', '0.75', '1.0'],
        'thresholds/0.0':'model_types',
        'thresholds/0.25': 'model_types',
        'thresholds/0.50': 'model_types',
        'thresholds/0.75': 'model_types',
        'thresholds/1.0': 'model_types'
        # 'thresholds':['Accuracy','F1_Score','F2_Score']
        
        }
    
    
    if not os.path.exists(folder):
        os.makedirs(folder)
        for path in ['data','models/XGB','predictions','train_test_split','images','thresholds',
                     'images/Accuracy','images/F1_Score','images/F2_Score','thresholds/0.0',
                     'thresholds/0.25','thresholds/0.50','thresholds/0.75','thresholds/1.0']:
            subfolder_path = os.path.join(folder, path)
            if verbose:
                print(subfolder_path)
            if subfolder_dict[path]:
                create_subfolders(subfolder_path, subfolder_dict[path])
            else:
                os.makedirs(subfolder_path)
            
    else:
        print("Directory Already Exists")

def get_max_index(data_path,prefix):
    """
    Get the count of the processed patient csv files
    
    Parameters:
    data_path (string): the filepath of the data
    prefix (string): the pattern of the data files of interest
    """
    # get all files in the datapath directory
    files = os.listdir(data_path)
    # set counter
    max_i = 0
    # for each file in the directory
    for file in files:
        # collect files with the prefix
        if prefix in file:
            # split name to get the numeric index
            i = file.split('_')[-1].split('.')[0]
            if i.isdigit():
                i = int(i)
                # iteratively update the max index 
                max_i = i if i > max_i else max_i
    return max_i

def create_filepath(config, subfolder, filename, **kwargs):
    """
    Creates a file path based on the configuration and provided keyword arguments.
    
    Parameters:
    config (dict): Configuration dictionary containing file path templates.
    subfolder (str): Key to specify the subfolder path in the configuration dictionary.
    filename (str): Specific file name in the configuration directory.
    **kwargs: Additional keyword arguments for placeholders in the file path template.

    Returns:
    str: Formatted file path.
    """
    if subfolder not in config['DIR']:
        raise ValueError(f"Subfolder '{subfolder}' not found in configuration.")

    file_dir_template = config['DIR'][subfolder]
    
    try:
        file_dir = file_dir_template.format(**kwargs)
        filepath = os.path.join(file_dir,filename)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir,exist_ok=True)
    except KeyError as e:
        missing_key = e.args[0]
        raise KeyError(f"Missing '{missing_key}' in keyword arguments for path formatting.") from None
    
    return filepath



