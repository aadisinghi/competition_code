
import pandas as pd
import numpy as np
import pickle
import math

def analysis_train(df):
    """
    series_id      int64
    time_step      int32
    close        float32
    volume       float32

    dtype: object
    """

    stats = df.groupby("series_id")["close"].agg(
        mean_close="mean",
        std_close="std"
    ).reset_index()

    stats.to_csv('analysis_data/training_data_analysis.csv',index=None)

def analysis_test(df):
    """
    window_id      int64
    time_step       int8
    close        float32
    volume       float32
    dtype: object

    """

    stats = df.groupby("window_id")["close"].agg(
        mean_close="mean",
        std_close="std"
    ).reset_index()

    stats.to_csv('analysis_data/testing_data_analysis.csv',index=None)


if __name__=='__main__':
    train = 'data/train.pkl'
    test = 'data/x_test.pkl'
    local_test = 'data/y_test_local.pkl'

    print("Loading dfs ------------------------------")
    train_df = pd.read_pickle(train)
    test_df = pd.read_pickle(test)
    local_test_df = pd.read_pickle(local_test)

    print("Generating analysis files ----------------------")
    analysis_train(train_df)
    analysis_test(test_df)



