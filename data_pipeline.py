from functions import *
import time 
import datetime

print("starting data pipeline at  ", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("-"*100)

""" Execute the full pipeline sequentially."""

# # Step 1: Extract data
# file_list = ["cowA_therm-0.csv", "cowA_therm-2.csv", "cowB_therm-0.csv", "cowB_therm-2.csv", "cowA_gas.csv", "cowB_gas.csv"]

# t0 = time.time()
# create_dataframe(file_list) # data extraction function for files list
# t1 = time.time()
# print("Step 1: Done")
# print("--------> Data Extracted in ", str(t1-t0), "seconds", "\n")


# Step 1 : Extract and Transform data
t0 = time.time()
clean_data() # transform data
t1 =  time.time()
print("Step 1: Done")
print("--------> Data extracted from computer storage and transformed in ", str(t1-t0), "seconds", "\n")


# Step 2: Peak Detection
t0 = time.time()
detect_peaks() # peak detection function
t1 = time.time()
print("Step 2: Done")
print("--------> Peaks detected in ", str(t1-t0), "seconds", "\n")

# Step 3: Merging data

t0 = time.time()
merge_dataframes() # merging thermstor and gas data function
t1 = time.time()
print("Step 3: Done")
print("--------> Data merged in ", str(t1-t0), "seconds", "\n")

# Step 4: Predicting Eructation
t0  = time.time()
train_xgboost_on_dataframe() # predicting with xgboost
t1 = time.time()
print("Step 4: Done")
print("--------> prediction made in ", str(t1-t0), "seconds", "\n")
