"""
Preprocessing module for Uber-Pickups dataset.
Handles data ingestion, consolidation, feature extraction, and temporal filtering.
"""
import pandas as pd
import glob
import os

def load_and_consolidate_data(folder_path: str) -> pd.DataFrame:
    """
    Load all monthly Uber raw data CSVs from a directory, consolidate them, 
    parse datetime, and extract temporal features.
    
    Parameters:
    - folder_path (str): Path to the directory containing the CSVs.
    
    Returns:
    - pd.DataFrame: A single consolidated and cleaned dataframe.
    """
    search_pattern = os.path.join(folder_path, "uber-raw-data-*14.csv")
    all_files = glob.glob(search_pattern)
    
    if not all_files:
        raise FileNotFoundError(f"No files matching the pattern were found in {folder_path}")
    
    df_list = []
    for file in all_files:
        print(f"Loading: {os.path.basename(file)}...")
        temp_df = pd.read_csv(file)
        df_list.append(temp_df)
        
    print("Consolidating datasets...")
    df = pd.concat(df_list, ignore_index=True)
    
    print("Parsing datetime and extracting features...")
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%m/%d/%Y %H:%M:%S')
    
    # Extract temporal features
    df['Month'] = df['Date/Time'].dt.month
    df['Month_Name'] = df['Date/Time'].dt.month_name()
    df['Hour'] = df['Date/Time'].dt.hour
    df['DayOfWeek'] = df['Date/Time'].dt.day_name()
    df['DayOfWeek_Num'] = df['Date/Time'].dt.dayofweek
    
    # Remove records missing crucial spatial coordinates
    df = df.dropna(subset=['Lat', 'Lon'])
    
    print(f"Data consolidation complete. Total rows: {len(df)}")
    return df

def filter_by_time(df: pd.DataFrame, day_name: str, hour: int) -> pd.DataFrame:
    """
    Filter the dataset to a specific day of the week and hour.
    
    Parameters:
    - df (pd.DataFrame): The preprocessed dataset.
    - day_name (str): Name of the day (e.g., 'Tuesday').
    - hour (int): Hour of the day (0-23).
    
    Returns:
    - pd.DataFrame: A filtered copy of the dataframe.
    """
    mask = (df['DayOfWeek'] == day_name) & (df['Hour'] == hour)
    return df[mask].copy()