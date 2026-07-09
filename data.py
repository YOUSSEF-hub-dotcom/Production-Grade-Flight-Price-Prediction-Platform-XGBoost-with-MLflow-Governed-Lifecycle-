import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("Data")

def run_data_pipeline(file_path):
    """
    Loads and preprocesses the raw flight database.
    Performs safe row-by-row transformations, avoiding data leakage.
    """
    logger.info("Loading dataset from: %s", file_path)
    df = pd.read_excel(file_path)
    
    logger.info("Starting initial data cleaning and validation...")
    
    # 1. Validate and convert DataType
    df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True)

    # 2. Extract hours and minutes from Dep_Time
    df["Dep_hour"] = pd.to_datetime(df["Dep_Time"], format='%H:%M').dt.hour
    df["Dep_minute"] = pd.to_datetime(df["Dep_Time"], format='%H:%M').dt.minute

    # 3. Extract hours and minutes from Arrival_Time
    # Splitting handling cases where arrival contains date string (e.g., '22:15 10 Mar')
    arrival_clean = df["Arrival_Time"].astype(str).str.split(' ').str[0]
    df["Arrival_hour"] = pd.to_datetime(arrival_clean, format='%H:%M').dt.hour
    df["Arrival_minute"] = pd.to_datetime(arrival_clean, format='%H:%M').dt.minute

    # 4. Convert Duration into minutes
    def convert_duration(duration):
        hours = 0
        minutes = 0
        parts = str(duration).split()
        for part in parts:
            if 'h' in part:
                hours = int(part.replace('h', ''))
            elif 'm' in part:
                minutes = int(part.replace('m', ''))
        return (hours * 60) + minutes

    df["Duration_mins"] = df["Duration"].apply(convert_duration)

    # 5. Convert Total_Stops to Numeric safely using replace to avoid turning unseen text to NaN
    stop_dict = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
    df['Total_Stops'] = df['Total_Stops'].replace(stop_dict)
    
    # NOTE: Total_Stops missing values imputation is moved to the split section 
    # to strictly avoid early Data Leakage using global mode.

    # 6. Drop redundant raw columns
    columns_to_drop = ['Dep_Time', 'Arrival_Time', 'Duration', 'Route', 'Additional_Info']
    df.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')

    # 7. Handle full row duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        logger.info("Removing %d duplicate rows from the dataset.", duplicate_count)
        df.drop_duplicates(inplace=True)

    # 8. Feature Engineering (Row-by-row deterministic features)
    logger.info("Executing row-by-row feature engineering...")
    df['Month_of_Journey'] = df['Date_of_Journey'].dt.month
    df['Days_of_Journey'] = df['Date_of_Journey'].dt.day
    df['Day_of_Week'] = df['Date_of_Journey'].dt.weekday
    df['Quarter'] = df['Date_of_Journey'].dt.quarter

    # Drop Date_of_Journey after extracting essential components
    df.drop("Date_of_Journey", axis=1, inplace=True)

    # Derived cyclical and domain-specific features
    df['is_weekend'] = df['Day_of_Week'].apply(lambda x: 1 if x >= 4 else 0)
    df['Path'] = df['Source'] + "-" + df['Destination']
    
    def assign_session(hour):
        if 4 <= hour < 8: return 'Early Morning'
        elif 8 <= hour < 12: return 'Morning'
        elif 12 <= hour < 16: return 'Noon'
        elif 16 <= hour < 20: return 'Evening'
        elif 20 <= hour < 24: return 'Night'
        else: return 'Late Night'

    df['Dep_Session'] = df['Dep_hour'].apply(assign_session)
    df['Is_Long_Flight'] = df['Duration_mins'].apply(lambda x: 1 if x > 480 else 0)
    df['is_peak_season'] = df['Month_of_Journey'].apply(lambda x: 1 if x in [3, 5, 6, 12] else 0)

    logger.info("Data pipeline completed successfully. Final shape: %s", str(df.shape))
    return df
