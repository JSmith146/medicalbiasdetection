import os
import pandas as pd
import numpy as np


# Set envrionmental variables
LOG_PATH = os.getenv('LOG_PATH')

# process patient demographic data
def process_demographic_data(data, config, facility="grady"):
    """
    Process the patient demographic data based on the specified medical facility.

    Parameters:
    data (DataFrame): The input patient data.
    facility (str): The name of the medical facility. Default is 'grady'.

    Returns:
    DataFrame: The processed demographic data.
    """
    # Ensure the input data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' must be a pandas DataFrame.")

    df = data.copy()
    
    # Process data for Grady facility
    if facility == 'grady':
        filepath = '/labs/kamaleswaranlab/MODS/Data/Grady_Data/1. Administrative Attributes/Encounters/'
        files = [file for file in os.listdir(filepath) if file.endswith('.txt')]
        
        data = []
        for file in files:
            year = file.split("_")[1]
            if len(year) > 4 or not year.isdigit():
                continue
            
            year = int(year)
            temp = df[df.year == year].copy()
            if len(temp) > 0:
                file_path = os.path.join(filepath, file)
                pat_df = pd.read_csv(file_path, sep='|', error_bad_lines=False, warn_bad_lines=False)
                temp = temp.merge(pat_df.drop(columns=['pat_id']), on=['csn'], copy=False)
                data.append(temp)

        if not data:
            return pd.DataFrame()

        demo_df = pd.concat(data, axis=0)

        drop_cols = ['age_y', 'hospital_discharge_date_time_y', 'discharge_status_y']
        rename_cols = {'hospital_discharge_date_time_x': 'hospital_discharge_date_time',
                       'discharge_status_x': 'discharge_status', 'age_x': 'age'}

        # Clean zipcode values
        demo_df['zip_code'] = demo_df['zip_code'].astype(str).apply(lambda x: x.split('-')[0])

        demo_df.drop(columns=drop_cols, inplace=True)
        demo_df.rename(columns=rename_cols, inplace=True)

        return demo_df

    elif facility == 'emory':
        # Add processing steps for Emory
        # Complete after linking Emory csns appropriately
        pass
    else:
        print("Medical facility does not exist")
        return pd.DataFrame()

    
# Helper functions to clean vitals signs and lab results
def clean_vitals(chunk, thresholds):
    """
    Clean and standardize the vitals data based on given thresholds.

    Parameters:
    chunk (DataFrame): The input vitals data.
    thresholds (dict): A dictionary of thresholds for vital signs.

    Returns:
    DataFrame: The cleaned vitals data.
    """
    if not isinstance(chunk, pd.DataFrame):
        raise TypeError("Input 'chunk' must be a pandas DataFrame.")
    if not isinstance(thresholds, dict):
        raise TypeError("Input 'thresholds' must be a dictionary.")

    process_cols = list(thresholds.keys())
    for feature in process_cols:
        chunk[feature] = chunk[feature].replace(r'\>|\<|\%|\/|\s', '', regex=True)
        chunk[feature] = pd.to_numeric(chunk[feature], errors='coerce')
        mask = (chunk[feature] < thresholds[feature][1]) & (chunk[feature] > thresholds[feature][0])
        chunk.loc[~mask, feature] = np.nan
    return chunk
    
    
def rolling_overlap(temp, window, variables):
    """
    Apply a rolling median with overlap to specified variables in the input data.

    Parameters:
    temp (DataFrame): The input data.
    window (int): The window size for rolling median.
    variables (list): List of variables to apply rolling median.

    Returns:
    DataFrame: The data with applied rolling median.
    """
    if not isinstance(temp, pd.DataFrame):
        raise TypeError("Input 'temp' must be a pandas DataFrame.")
    if not isinstance(variables, list):
        raise TypeError("Input 'variables' must be a list.")
    # if not isinstance(window, int) or not isinstance(overlap, int):
    #     raise TypeError("Input 'window' and 'overlap' must be integers.")

    rolled = temp.copy()
    rolled[variables] = rolled.rolling(window, min_periods = 1)[variables].aggregate("median")
    
    # input_data_frame[var_list]= input_data_frame[var_list].fillna(pd.rolling_mean(input_data_frame[var_list], 6, min_periods=1))
    
    # rolled[variables] = rolled[variables].fillna(rolled.rolling(window, min_periods = 1)[variables].aggregate('median'))
    rolled = rolled.reset_index(drop = True)
    return rolled

def nextClosestTime(time_series, t):
    """
    Find the closest time in the time_series to the given time t.

    Parameters:
    time_series (list or Series): The list of times to search.
    t (datetime or similar): The time to find the closest match for.

    Returns:
    datetime or similar: The closest time to t found in time_series.
    """
    if not (isinstance(time_series, list) or isinstance(time_series, pd.Series)):
        raise TypeError("Input 'time_series' must be a list or pandas Series.")

    diffVec = list(map(lambda x:np.abs(x-t),time_series))
    min_diff_idx = np.argmin(diffVec)
    return time_series[min_diff_idx]

def feature_informative_missingness(patient_vitals_df):
    """
    Enhance the dataset of patient vital signs with additional features that capture information about the missingness of the data
    Missingness refers to situations where the absence of data can itself be a meaninful factor
    Source: https://www.researchgate.net/publication/338628580
            
    
    Parameters:
    patient_vitals_df (DataFrame): patient vital sign information
    """
    # copy patient vital df
    patient = patient_vitals_df.copy()
    # collect original index of data
    idx_temp = patient.index
    # reset index to ensure sequential order
    patient = patient.reset_index(drop=True)
    # loop through each vital sign
    for col in patient.columns:
        # get index of nonmissing data points
        nonmissing_idx = patient.index[~patient[col].isna()].tolist()
        # generate column names for new information missingness features
        f1_name = f"{col}_interval_f1"
        f2_name = f"{col}_interval_f2"
        diff_name = f"{col}_diff"
        
        # create a sequential count of non-missing data points
        patient.loc[nonmissing_idx,f1_name] = np.arange(1,len(nonmissing_idx)+1)
        patient[f1_name] = patient[f1_name].ffill().fillna(0)

        v = (0+patient[col].isna()).replace(0,np.nan)
        cumsum = v.cumsum().fillna(method='pad')
        reset = -cumsum[v.isnull()].diff().fillna(cumsum)
        patient[f2_name] = v.where(v.notnull(), reset).cumsum().fillna(0)
        
        if nonmissing_idx==[]:
            patient.loc[:, f2_name] = -1
        else:
            patient.loc[:nonmissing_idx[0]-1, f2_name] = -1
        
        patient[diff_name] = patient.loc[nonmissing_idx, col].diff()
        patient[diff_name] = patient[diff_name].fillna(method = "ffill")
    return patient

def feature_slide_window(vital_df,window):
    """
    Generate rolling window features for a given dataframe of vital signs
    Parameters:
    vital_df (DataFrame): dataframe of patient vital sign data
    
    Returns:
    rollingArr (DataFrame): concatenated dataframe of patient vital sign data with rolling window data added
    """
    # identify the types of rolling window calculation
    roll_types= ['mean','median','min','max','std','diff']
    # calculate the difference between each row and the row following it
    diff = vital_df.shift(-1) - vital_df
    # create a list to store the results of rolling window calculations
    rollingArr = []
    for r_type in roll_types:
        if r_type =='diff':
            # calculate the standard deviation of differences calculated over 6 time periods
            r_type = 'std'
            rolling = diff.rolling(window,min_periods=1).aggregate(r_type).reset_index(drop = True)
            roll_cols = [f"d{r_type}_{x}" for x in vital_df.columns]
            rolling.columns = roll_cols
            rollingArr.append(rolling)
            continue
        # calculate the rollings for previous 6 time periods
        roll_cols = [f"{r_type}_{x}" for x in vital_df.columns]
        rolling = vital_df.rolling(window, min_periods = 1).aggregate(r_type).reset_index(drop = True)
        rolling.columns = roll_cols
        rollingArr.append(rolling)
    return pd.concat(rollingArr,axis=1)
        
        
def feature_empiric_score(temp):
    # get column names from temp dataframe
    cols = temp.columns
    # HEART RATE SCORING (NEWS2 metric)
    col = "pulse"
    if col in cols:
        temp["pulse_score"] = 0
        mask = (temp["pulse"] <= 40) | (temp["pulse"] >= 131)
        temp.loc[mask,"pulse_score"] = 3
        mask = (temp["pulse"] <= 130) & (temp["pulse"] >= 111)
        temp.loc[mask,"pulse_score"] = 2
        mask = ((temp["pulse"] <= 50) & (temp["pulse"] >= 41)) | ((temp["pulse"] <= 110) & (temp["pulse"] >= 91))
        temp.loc[mask,"pulse_score"] = 1
        temp.loc[temp["pulse"].isna(),"pulse_score"] = np.nan


    # TEMPERATURE SCORING (NEWS2 metric)
    col = "temperature"
    if col in cols:
        temp["temperature_score"] = 0
        mask = (temp["temperature"] <= 35)
        temp.loc[mask,"temperature_score"] = 3
        mask = (temp["temperature"] >= 39.1)
        temp.loc[mask,"temperature_score"] = 2
        mask = ((temp["temperature"] <= 36.0) & (temp["temperature"] >= 35.1)) | ((temp["temperature"] <= 39.0) & (temp["temperature"] >= 38.1))
        temp.loc[mask,"temperature_score"] = 1
        temp.loc[temp["temperature"].isna(),"temperature_score"] = np.nan


    # Resp Score (NEWS2 metric)
    col = "unassisted_resp_rate"
    if col in cols:
        temp["unassisted_resp_rate_score"] = 0
        mask = (temp["unassisted_resp_rate"] < 8) | (temp["unassisted_resp_rate"] > 25)
        temp.loc[mask,"unassisted_resp_rate_score"] = 3
        mask = ((temp["unassisted_resp_rate"] <= 24) & (temp["unassisted_resp_rate"] >= 21))
        temp.loc[mask,"unassisted_resp_rate_score"] = 2
        mask = ((temp["unassisted_resp_rate"] <=11) & (temp["unassisted_resp_rate"] >= 9))
        temp.loc[mask,"unassisted_resp_rate_score"] = 1

        temp.loc[temp["unassisted_resp_rate"].isna(),"unassisted_resp_rate_score"] = np.nan

    #MAP Score: (SOFA metric)
    col = "map_line"
    if col in cols:
        temp["map_line_score"] = 1
        mask = (temp["map_line"] >= 70)
        temp.loc[mask, "map_line_score"] = 0
        temp.loc[temp["map_line"].isna(),"map_line_score"] = np.nan
    
    # Creatinine score: (SOFA metric)
    col = "creatinine"
    if col in cols:
        temp["creatinine_score"] = 4
        mask = (temp["creatinine"]< 5)
        temp.loc[mask, "creatinine_score"] = 3
        mask = (temp["creatinine"] < 3.5)
        temp.loc[mask, "creatinine_score"] = 2
        mask = (temp["creatinine"] < 2)
        temp.loc[mask, "creatinine_score"] = 1
        mask = (temp["creatinine"] < 1.2)
        temp.loc[mask, "creatinine_score"] = 0
        temp.loc[temp["creatinine"].isna(),"creatinine_score"] = np.nan


    # qsofa:
    # col = "qsofa"
    # if col in cols:
    temp["qsofa"] = 0
    mask = (temp["sbp_line"] <= 100) & (temp["unassisted_resp_rate"] >= 22)
    temp.loc[mask, "qsofa"] = 1
    mask = (temp["sbp_line"].isna()) | (temp["unassisted_resp_rate"].isna())
    temp.loc[mask, "qsofa"] = np.nan

    # # sirs
    # col = "sirs"
    # if col in cols:
    #     temp["sirs"] = 0
    #     mask_temp = temp["temperature"] > 38
    #     mask_pulse = temp["pulse"] > 90
    #     mask_res_rate = ((temp["unassisted_resp_rate"] > 20)|(temp["partial_pressure_of_carbon_dioxide_(paco2)"]<32))
    #     mask = (mask_temp+mask_pulse+mask_res_rate)>1
    #     temp.loc[mask,"sirs"] = 1
    #     mask = (temp["temperature"].isna())|(temp["pulse"].isna())|(temp["unassisted_resp_rate"].isna())|(temp["partial_pressure_of_carbon_dioxide_(paco2)"].isna())
    #     temp.loc[mask,"sirs"] = np.nan
    
    
    # Platelets score: (SOFA Metric)
    col = "platelets"
    if col in cols:
        temp["platelets_score"] = 0
        mask = (temp["platelets"] <= 150)
        temp.loc[mask, "platelets_score"] = 1
        mask = (temp["platelets"] <= 100)
        temp.loc[mask, "platelets_score"] = 2
        mask = (temp["platelets"] <= 50)
        temp.loc[mask, "platelets_score"] = 3
        mask = (temp["platelets"] <= 20)
        temp.loc[mask, "platelets_score"] = 4

        temp.loc[temp["platelets"].isna(),"platelets_score"] = np.nan


        
     # Bilirubin score: (SOFA Metric)
    col = "bilirubin_total"
    if col in cols:
        
        temp["bilirubin_score"] = 4
        mask = (temp["bilirubin_total"] < 12)
        temp.loc[mask, "bilirubin_score"] = 3
        mask = (temp["bilirubin_total"] < 6)
        temp.loc[mask, "bilirubin_score"] = 2
        mask = (temp["bilirubin_total"] < 2)
        temp.loc[mask, "bilirubin_score"] = 1
        mask = (temp["bilirubin_total"] < 1.2)
        temp.loc[mask, "bilirubin_score"] = 0
        temp.loc[temp["bilirubin_total"].isna(),"bilirubin_score"] = np.nan
    
    return(temp)

def update_patient_type(dataframe):
    df = dataframe.copy()
    df['PATIENT_TYPE_3'] = ''
    add_jail = ['fulton county jail','901 rice','dekalb county jail','4425 memorial','clayton county jail','9157 tara blvd']
    
    mask = df['ADD_LINE_1'].str.lower().isin(add_jail)
    df.loc[mask,'PATIENT_TYPE_3'] = 'Prisoner'
    
    mask = df['ADD_LINE_2'].str.lower().isin(add_jail)
    df.loc[mask,'PATIENT_TYPE_3'] = 'Prisoner'
    
    types = ['Prisoner','Homeless','Transgender']
    for t in types:
        df[t] = (df == t).any(axis=1).astype(int)
        
    return df

def update_patient_language(dataframe):
    df = dataframe.copy()
    langs = [
        'English','Spanish','French','Amharic','Burmese','Bengali',
        'Vietnamese','Hindi','Arabic','Sign Language','Portuguese', 
        'Wolof','ahmaric','Igbo (Ibo)','Haitian Creole','Indonesian',
        'Romanian', 'Nepali'
            ]
    
    df.loc[~df['LANGUAGE'].isin(langs), 'LANGUAGE'] = 'Other'
    return df

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




