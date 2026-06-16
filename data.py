import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger("Data")


def run_data_pipeline(file_path):
    df = pd.read_excel(file_path)
    logger.info("Loading Dataset ....")
    pd.set_option('display.width', None)

    print(df.head(30))

    logger.info("=========== Basic Functions ==========")
    logger.info("information about data:")
    print(df.info())

    logger.info("Statistical Operations:")
    print(df.describe())

    logger.info("Columns:")
    print(df.columns)

    logger.info("number of rows & columns:")
    print(df.shape)

    logger.info("Column types:")
    print(df.dtypes)

    logger.info("=========== Data Cleaning ==========")

    logger.info("Validate DataType:")
    df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True)

    # extract hours and minute from Dep_Time
    df["Dep_hour"] = pd.to_datetime(df["Dep_Time"], format='%H:%M').dt.hour
    df["Dep_minute"] = pd.to_datetime(df["Dep_Time"], format='%H:%M').dt.minute

    # extract hours and minute from Arrival_Time
    df["Arrival_hour"] = pd.to_datetime(df["Arrival_Time"].str.split(' ').str[0], format='%H:%M').dt.hour
    df["Arrival_minute"] = pd.to_datetime(df["Arrival_Time"].str.split(' ').str[0], format='%H:%M').dt.minute

    # Convert Duration just to minutes
    def convert_duration(duration):
        hours = 0
        minutes = 0
        parts = duration.split()
        for part in parts:
            if 'h' in part:
                hours = int(part.replace('h', ''))
            elif 'm' in part:
                minutes = int(part.replace('m', ''))
        return (hours * 60) + minutes

    df["Duration_mins"] = df["Duration"].apply(convert_duration)

    # Convert Total_Stops to Numeric
    stop_dict = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
    df['Total_Stops'] = df['Total_Stops'].map(stop_dict)

    # Drop raw columns
    df.drop(['Dep_Time', 'Arrival_Time', 'Duration', 'Route', 'Additional_Info'], axis=1, inplace=True)

    logger.info("number of frequency rows")
    print(df.duplicated().sum())

    # Delete duplicates
    df.drop_duplicates(inplace=True)
    print(f"Dataset shape after removing duplicates: {df.shape}")

    logger.info("missing values:")
    print(df.isnull().sum())

    mode_val = df["Total_Stops"].mode()[0]
    df["Total_Stops"] = df["Total_Stops"].fillna(mode_val)
    logger.info(f"Filled missing Total_Stops with mode: {mode_val}")

    # Feature Engineering (نقلناها هنا لتكتمل الـ Pipeline الأساسية بالـ Features الجديدة)
    df['Month_of_Journey'] = df['Date_of_Journey'].dt.month
    df['Days_of_Journey'] = df['Date_of_Journey'].dt.day
    df['Day_of_Week'] = df['Date_of_Journey'].dt.weekday
    df['Quarter'] = df['Date_of_Journey'].dt.quarter

    if df['Date_of_Journey'].dt.year.nunique() <= 1:
        print("Year column ignored because it has only one value.")

    df.drop("Date_of_Journey", axis=1, inplace=True)

    df['is_weekend'] = df['Day_of_Week'].apply(lambda x: 1 if x >= 4 else 0)
    df['Path'] = df['Source'] + "-" + df['Destination']
    
    def assign_session(hour):
        if (hour >= 4) and (hour < 8): return 'Early Morning'
        elif (hour >= 8) and (hour < 12): return 'Morning'
        elif (hour >= 12) and (hour < 16): return 'Noon'
        elif (hour >= 16) and (hour < 20): return 'Evening'
        elif (hour >= 20) and (hour < 24): return 'Night'
        else: return 'Late Night'

    df['Dep_Session'] = df['Dep_hour'].apply(assign_session)
    df['Is_Long_Flight'] = df['Duration_mins'].apply(lambda x: 1 if x > 480 else 0)
    df['is_peak_season'] = df['Month_of_Journey'].apply(lambda x: 1 if x in [3, 5, 6, 12] else 0)

    print(df.head(30))
    return df

