#Importing important libraries

import pandas as pd
# import numpy as np
# import seaborn as sns
# sns.__version__
import matplotlib.pyplot as plt
import  os 
import scipy
from scipy.signal import find_peaks
import sklearn
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score
import csv




# Data Cleaning pipeline
def clean_data():

    """ 
    Renaming "raw" column for better clarity
    Handling Missing Data and Duplicates
    Conversion of data types and timestamps
    Checks for outlier 
      
    Parameters:
    - df: Dataframe  to clean
    - file_type: Dataframe category  either gas_files or thermistor files

                                    """
    dataframes = {}
    data_dir = "./data/"

    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(data_dir, filename)
            df_name =  "df_" + filename.replace(".csv", "").replace("-","_")
            df = pd.read_csv(filepath)

            dataframes[df_name] = df

    clean_dataframes = {}

    for name, df in dataframes.items():
        #Dropping Missing Data
        df = df.dropna()

        #Dropping duplicates if any present
        df =  df.drop_duplicates()

        #Converting time to datetime format
        if  "time_ms" in df.columns or "epoch_ms" in df.columns:
            time_col = "time_ms" if "time_ms" in df.columns else "epoch_ms" # checks if column is time_ms colum or epoch_ms 
            df['timestamp'] = pd.to_datetime(df[time_col], unit='ms').dt.strftime('%H:%M%:%S.%f') #convert to date time format and extract just time
            df.drop(time_col, axis =1, inplace =True) # dropping pre-exisitng time column
            df =  df[['timestamp'] + [col for col in df.columns if col != 'timestamp']]

        #Converting of data type:
        for col, dtype in {"thermistor_id": int, "temperature": float, "raw": float,
                           "co2": float, "co2temp": float, "ch4": float, "ch4temp": float}.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

        #Renaming "raw" column to "therm_voltage_ratio", "co2"  column to "co2_conc", 'ch4', to 'ch4_conc'
        df.rename(columns = {"raw": "therm_voltage_ratio","co2": "co2_conc", "ch4" : "ch4_conc"}, inplace =True)
        
    
        #Checks for negative values and dropping negative rows
        df = df[(df.select_dtypes(include="number") > 0).any(axis =1)] # removing outlier temperatures less than zero

        #sorting by increasing time for better clarity
        df =  df.sort_values(by =['timestamp'])

        # Resetting index
        df = df.reset_index(drop= True) #to avoid addition of old index as column
        clean_dataframes[name] = df

    save_data(clean_dataframes, save_dir= "./data/processed/cleaned") # Save data in file directory





#Peak Detection pipeline
def detect_peaks():
    save_dir = "./data/processed/cleaned"
    clean_dataframes = {}

    for filename in os.listdir(save_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(save_dir, filename)
            df_name =  filename.replace(".csv", "")
            df = pd.read_csv(filepath)

            clean_dataframes[df_name] = df

    for key, df in clean_dataframes.items():

        df['timestamp'] = df['timestamp'].str.strip()

        # Fixing malformed entries first (e.g., remove % if possible)
        df['timestamp'] = df['timestamp'].str.replace('%', ':', regex=False)
        
        #converting timestamps to date-time format
        # Now parsing using correct expected format
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S.%f', errors='coerce')


        if 'gas' in key:
            peak_height =  df['ch4_conc'].mean() + 2*df['ch4_conc'].std() #calculating peak height based on human judgement and visualization of the methan concentration over tim

            
            peak_idx, _ = find_peaks(df['ch4_conc'], height=peak_height, distance= 110, prominence=0.05)

            print(f"Peaks detected for {key}: {len(peak_idx)} at indices {peak_idx}")

            #adding peak column to datafame
            df['peak'] = 0  # Initializing the 'peak' column with 0 (no peak)
            df.loc[peak_idx, 'peak'] = 1

            clean_dataframes[key] = df #updating in dataframe
            # peak_data = df.iloc[peak_idx]

            print(f"The plot of methane concentration against time for {key}")

            plt.figure(figsize=(20,12))
            plt.plot(df['timestamp'], df['ch4_conc'], label = "Methane Concentration (ppm)", color = 'blue')
            plt.scatter(df["timestamp"].iloc[peak_idx], df["ch4_conc"].iloc[peak_idx], color = "red", alpha=0.6, label = 'Peaks')
            plt.xlabel("Time in milliseconds")
            plt.ylabel("Methane Concentration")
            plt.title(f"Methane Concentration Peaks for {key}")
            plt.grid(True)
            plt.legend()
            plt.show()


    save_data(clean_dataframes, save_dir= "./data/processed/peaks")
        

#merging dataframes
def  merge_dataframes():

    save_dir = "./data/processed/peaks" #with peak column added
    clean_dataframes = {}

    for filename in os.listdir(save_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(save_dir, filename)
            df_name =  filename.replace(".csv", "")
            df = pd.read_csv(filepath)

            clean_dataframes[df_name] = df

    merged_dataframes = {}
    merged_pairs = [ ("df_cowA_gas", "df_cowA_therm_0"),
        ("df_cowA_gas", "df_cowA_therm_2"),
        ("df_cowB_gas", "df_cowB_therm_0"),
        ("df_cowB_gas", "df_cowB_therm_2"),
                                    ] #list of tuples

    for gas_key, therm_key in merged_pairs:
    
        if gas_key in clean_dataframes.keys() and therm_key in clean_dataframes.keys():

            # Convert timestamp column to datetime
            clean_dataframes[therm_key]["timestamp"] = pd.to_datetime(clean_dataframes[therm_key]["timestamp"])
            clean_dataframes[gas_key]["timestamp"] = pd.to_datetime(clean_dataframes[gas_key]["timestamp"])

            # Ensure they are sorted by timestamp
            clean_dataframes[therm_key] = clean_dataframes[therm_key].sort_values(by="timestamp")
            clean_dataframes[gas_key] = clean_dataframes[gas_key].sort_values(by="timestamp")

            merged_df = pd.merge_asof(clean_dataframes[therm_key], clean_dataframes[gas_key], on = "timestamp", direction = "nearest")
            merged_dataframes[f'merged_{gas_key}_{therm_key}'] = merged_df

    save_data(merged_dataframes, save_dir= "./data/processed/merged_data")




#Predicting eructation
def train_xgboost_on_dataframe():
    """
    Training an XGBoost model on a single time-series DataFrame using a chronological train-test split.

    Parameters:
    - merged_df:  DataFrame to process and train on.
    - train_size: Fraction of data used for training 
   
    Returns:
    - accuracy: The accuracy score of the model on the test set.
    """
    train_size=0.8
    save_dir = "./data/processed/merged_data"
    merged_dataframes = {}
    model_eval_accuracy= {}

    for filename in os.listdir(save_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(save_dir, filename)
            # df_name =  "df_" + filename.replace(".csv", "").replace("-","_")
            df_name =  filename.replace(".csv", "")
            df = pd.read_csv(filepath)

            merged_dataframes[df_name] = df

    for df_name, merged_df in merged_dataframes.items():
        # Convert 'timestamp' column to datetime
        
        merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])

        
        # Boundary
        print("\n" + "="*50)
        print(f"Training and testing on DataFrame: {df_name}")
        print("="*50)
        
        #additional features
        # peak_df = merged_df[merged_df['peak'] ==1]
        # peak_time = peak_df['timestamp']
        merged_df['time_since_last_peak'] = merged_df['timestamp'] - merged_df['timestamp'].where(merged_df['peak'] ==1).ffill()
        merged_df['time_since_last_peak'] = merged_df['time_since_last_peak'].dt.total_seconds().fillna(0)
        
        # Rolling features (over the past 10 observations)
        merged_df['ch4_rolling_mean'] = merged_df['ch4_conc'].rolling(window=10).mean().fillna(0)
        merged_df['co2_rolling_mean'] = merged_df['co2_conc'].rolling(window=10).mean().fillna(0)

        # Renaming 'peak' column to 'eructation'
        merged_df = merged_df.rename(columns={'peak': 'eructation'})  
        
        # Sorting by timestamp for time-series order
        merged_df = merged_df.sort_values(by=['timestamp'], ascending=True)  
        
        #time difference in seconds from the first timestamp
        merged_df['time_diff_seconds'] = (merged_df['timestamp'] - merged_df['timestamp'].min()).dt.total_seconds()

        # Dropping unnecessary features to avoid data leakage
        merged_df = merged_df.drop(columns=['ch4_conc', 'ch4temp', 'thermistor_id'])

        # Feature selection
        X = merged_df.drop(columns=['eructation', 'timestamp'])
        y = merged_df['eructation']
        
        # Train-test split index
        split_idx = int(len(merged_df) * train_size)
        
        # Train-test split (chronological to avoid data leakage)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        #Initializing StandardScaler
        scaler = StandardScaler()

        # scaling
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initializing XGBoost Classifier
        model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
        
        # Training model
        model.fit(X_train_scaled, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test_scaled)
        
        # obtaining accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")

        model_eval_accuracy[df_name] =  accuracy

    test_result(df_name, model_eval_accuracy)

        


def  save_data(dataframes, save_dir = "./data/processed/"):

    os.makedirs(save_dir, exist_ok=True)

    for name , df in dataframes.items():
        file_path = os.path.join(save_dir, f"{name}.csv" )
        df.to_csv(file_path, index=False)
        print(f"Saved processed {name}.csv successfully")


def test_result(df_name, model_eval_accuracy):

    output_dir = "./data/processed/model_evaluation"
    os.makedirs(output_dir, exist_ok=True)  # Creates directories if they don’t exist

    with open(f"{output_dir}/model_eval_accuracy.csv", "w", newline="") as fp:
        
        #creating a writer object
        writer = csv.DictWriter(fp, fieldnames=model_eval_accuracy.keys())

        # Writing the header row
        writer.writeheader()
        writer.writerow(model_eval_accuracy)

        #write the data rows
        writer.writerow(model_eval_accuracy)
        print('Done writing dict to a csv file')
        





