import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger("Data")

def run_data_pipeline(file_path):
    """
    Loads and preprocesses the raw flight dataset.
    Executes tasks in the following strict architectural pipeline:
    1. Quick Data Inspection
    2. Basic Data Cleaning
    3. Check Skew & Outliers (Including Correlation Analysis)
    4. Feature Engineering
    """
    logger.info("Loading Dataset ....")
    df = pd.read_excel(file_path)
    pd.set_option('display.width', None)

    # ====================================================
    # STEP 1: Quick Data Inspection
    # ====================================================
    logger.info("=========== STEP 1: Quick Data Inspection ==========")
    
    print(df.head(30))

    logger.info("Information about data:")
    print(df.info())

    logger.info("Statistical Operations:")
    print(df.describe())

    logger.info("Columns:")
    print(df.columns)

    logger.info("Number of rows & columns:")
    print(df.shape)

    logger.info("Column types:")
    print(df.dtypes)


    # ====================================================
    # STEP 2: Basic Data Cleaning
    # ====================================================
    logger.info("=========== STEP 2: Basic Data Cleaning ==========")

    logger.info("Validate DataType:")
    df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True)

    # Extract hours and minutes from Dep_Time
    df["Dep_hour"] = pd.to_datetime(df["Dep_Time"], format='%H:%M').dt.hour
    df["Dep_minute"] = pd.to_datetime(df["Dep_Time"], format='%H:%M').dt.minute

    # Extract hours and minutes from Arrival_Time
    arrival_clean = df["Arrival_Time"].astype(str).str.split(' ').str[0]
    df["Arrival_hour"] = pd.to_datetime(arrival_clean, format='%H:%M').dt.hour
    df["Arrival_minute"] = pd.to_datetime(arrival_clean, format='%H:%M').dt.minute

    # Convert Duration into minutes
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

    # Convert Total_Stops to Numeric safely using replace to avoid turning unseen text to NaN
    stop_dict = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
    df['Total_Stops'] = df['Total_Stops'].replace(stop_dict)
    
    # NOTE: Total_Stops missing values imputation is handled downstream inside model.py 
    # to strictly avoid early Data Leakage via global mode computation.

    # Drop raw columns
    columns_to_drop = ['Dep_Time', 'Arrival_Time', 'Duration', 'Route', 'Additional_Info']
    df.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')

    logger.info("Number of frequency rows:")
    print(df.duplicated().sum())

    # Delete duplicates
    df.drop_duplicates(inplace=True)
    print(f"Dataset shape after removing duplicates: {df.shape}")

    logger.info("Missing values:")
    print(df.isnull().sum())


    # ====================================================
    # STEP 3: Check Skew & Outliers (Including Correlation)
    # ====================================================
    logger.info("=========== STEP 3: Check Skew & Outliers ==========")
    
    if 'Price' in df.columns:
        # 1. Target Skewness Analysis
        skew_value = df['Price'].skew()
        logger.info(f"Baseline Skew Value of Price: {skew_value:.4f}")
        
        plt.figure(figsize=(8,5))
        sns.histplot(df['Price'], kde=True, color='blue')
        plt.title("Distribution of Price (Baseline Skew)")
        plt.xlabel("Price")
        plt.ylabel("Frequency")
        plt.show()
        plt.close()
        
        # 2. Outlier Detection via Interquartile Range (IQR)
        Q1 = df['Price'].quantile(0.25)
        Q3 = df['Price'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df['Price'] < lower_bound) | (df['Price'] > upper_bound)]
        logger.info(f"Outliers Detected via IQR: {len(outliers)} rows")
        logger.info(f"Percentage of Outliers: {(len(outliers) / len(df) * 100):.4f} %")
        
        plt.figure(figsize=(8,4))
        sns.boxplot(x=df['Price'], color='red')
        plt.title("BoxPlot for Outlier Detection in Price")
        plt.xlabel("Price")
        plt.show()
        plt.close()

        # 3. Correlation Analysis (Pearson Correlation for Linear Relations)
        logger.info("Executing Pearson Correlation Analysis...")
        # تحديد الأعمدة الرقمية الحالية لحساب الارتباط مع السعر
        numerical_cols = ['Price', 'Duration_mins', 'Total_Stops', 'Dep_hour', 'Dep_minute', 'Arrival_hour', 'Arrival_minute']
        existing_num_cols = [col for col in numerical_cols if col in df.columns]
        
        pearson_corr = df[existing_num_cols].corr(method='pearson')
        logger.info("\nPearson Correlation Matrix:\n%s", str(pearson_corr))

        plt.figure(figsize=(10, 8))
        sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Pearson Correlation Heatmap (Numerical Features & Price)")
        plt.tight_layout()
        plt.show()
        plt.close()
        
    else:
        logger.warning("Target column 'Price' not found. Skipping Skew, Outlier, and Correlation Check.")


    # ====================================================
    # STEP 4: Feature Engineering
    # ====================================================
    logger.info("=========== STEP 4: Feature Engineering ==========")
    
    # Extract temporal details from Date_of_Journey
    df['Month_of_Journey'] = df['Date_of_Journey'].dt.month
    df['Days_of_Journey'] = df['Date_of_Journey'].dt.day
    df['Day_of_Week'] = df['Date_of_Journey'].dt.weekday
    df['Quarter'] = df['Date_of_Journey'].dt.quarter

    if df['Date_of_Journey'].dt.year.nunique() <= 1:
        print("Year column ignored because it has only one value.")

    # Drop the original date column after extraction
    df.drop("Date_of_Journey", axis=1, inplace=True, errors='ignore')

    # Build advanced binary, categorical, and interaction features
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

    # Final look at the processed dataset
    print(df.head(30))
    logger.info("Data pipeline completed successfully. Final shape: %s", str(df.shape))
    
    return df
