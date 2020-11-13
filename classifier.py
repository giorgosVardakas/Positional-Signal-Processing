# Libraries
import numpy as np
import pandas as pd

from seglearn.pipe import Pype
from seglearn.transform import FeatureRep, Segment
from seglearn.feature_functions import mean, var, std, skew
from seglearn.base import TS_Data

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, GroupKFold

import matplotlib.pyplot as plt

import sys
import pdb

# Global variables
data_path = "./Data/HomoreDataFromVariousActivities/"

def read_dataset():
    activities_path = data_path + "activities.csv"
    accelerometer_path = data_path + "accelerometer_various_activities_10hz.csv"
    gyroscope_path = data_path + "gyroscope_various_activities_10hz.csv"
    heartrate_path = data_path + "heartrate_various_activities.csv"

    df_activities = pd.read_csv(activities_path)
    df_accelerometer = pd.read_csv(accelerometer_path)
    df_gyroscope = pd.read_csv(gyroscope_path)
    df_heartrate = pd.read_csv(heartrate_path)

    # Change timestamp format from string to datetime
    datetime_format = "%Y-%m-%d %H:%M:%S.%f"
    df_accelerometer["TIMESTAMP"] = pd.to_datetime(df_accelerometer["TIMESTAMP"], format=datetime_format)
    df_gyroscope["TIMESTAMP"] = pd.to_datetime(df_gyroscope["TIMESTAMP"], format=datetime_format)
    df_heartrate["TIMESTAMP"] = pd.to_datetime(df_heartrate["TIMESTAMP"], format=datetime_format)

    return df_activities, df_accelerometer, df_gyroscope, df_heartrate

def drop_lying_on_data(df_activities, df_accelerometer, df_gyroscope, df_heartrate):
    lying_on_id = df_activities.loc[df_activities["NAME"] == "lying on"]["ACTIVITY_ID"].iloc[0]

    # Droping the rows
    df_accelerometer.drop(df_accelerometer[df_accelerometer["ACTIVITY_ID"] == lying_on_id].index, inplace=True)
    df_gyroscope.drop(df_gyroscope[df_gyroscope["ACTIVITY_ID"] == lying_on_id].index, inplace=True)
    df_heartrate.drop(df_heartrate[df_heartrate["ACTIVITY_ID"] == lying_on_id].index, inplace=True)

    # Reseting the index
    df_accelerometer.reset_index(inplace=True)
    df_gyroscope.reset_index(inplace=True)
    df_heartrate.reset_index(inplace=True)

def find_activity_changes(df_accelerometer):
    #Only for df_accelerometer and for df_gyroscope
    # Sampling frequency of accelerometer (10 Hz)
    sampling_frequency = df_accelerometer.loc[2, "TIMESTAMP"] - df_accelerometer.loc[1, "TIMESTAMP"]
    # Finding the indexs where the samples differ more than sampling frequency
    activity_cutoff = df_accelerometer.loc[df_accelerometer['TIMESTAMP'] - df_accelerometer['TIMESTAMP'].shift() > sampling_frequency]

    return activity_cutoff.index

def main(argv):
    #pdb.set_trace()
    df_activities, df_accelerometer, df_gyroscope, df_heartrate = read_dataset()

    drop_lying_on_data(df_activities, df_accelerometer, df_gyroscope, df_heartrate)
    print(df_accelerometer.shape, df_gyroscope.shape, df_heartrate.shape)

    activity_cutoff = find_activity_changes(df_accelerometer)
    #pdb.set_trace()

    features = {"mean":mean, "var":var, "std":std, "skew":skew}
    transformation_pype = Pype([
        ("segment", Segment(width=100, overlap=0.2)),
        ("features", FeatureRep(features = features)),
        ("scaler", StandardScaler())
        #("rf_clf", RandomForestClassifier())
    ], memory=None)

    prev_index = 0
    for index in activity_cutoff:
        print(df_accelerometer.loc[prev_index : index - 1])
        prev_index = index
        #print(index)

if __name__ == "__main__":
	main(sys.argv[1:])
