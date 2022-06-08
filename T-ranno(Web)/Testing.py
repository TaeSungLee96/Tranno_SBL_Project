
from flask import Blueprint, request, jsonify
from flask import Flask 
import io                                                         # 파일 경로 접근시 필요한 python 내장 라이브러리
import numpy as np
import pandas as pd
from flask import Flask                                          # python web framework
from flask import Flask, make_response, request                                     # 웹 요청 관련 모듈
from flask import render_template, redirect, url_for          # flask에서 필요한 모듈
from flask import jsonify                                        # import JSON을 해도되지만 여기서는 flask 내부에서 지원하는 jsonify를 사용
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import Sequence                                # 이 모듈이 없으면 사용자가 만든 generator에서 'shape'가 없다고 하는 에러가 발생할 수 있음
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
import tensorflow as tf
import csv
from scipy import stats
import sklearn as sk
import scipy.stats
import os
import requests
import time


app = Flask(__name__)

@app.route('/testing', methods=['POST'])
def signup_post():
    user = request.get_json()
    response = {
                'variable1' : user['Testing input_1'],
                'variable2' : user['Testing input_2'],
                'variable3' : user['Testing input_3'],
                'variable4' : user['Testing input_4'],
                'variable5' : user['Testing input_5'],
                'variable6' : user['Testing input_6'],
                'variable7' : user['Testing input_7']
                }

    model_input_ = np.array([[response['variable1'],response['variable2'],response['variable3'],response['variable4'],response['variable5'],response['variable6'],response['variable7']]])
    model_input_t_ = pd.DataFrame(model_input_, columns=['variable1', 'variable2', 'variable3', 'variable4', 'variable5', 'variable6','variable7'])
    model_input = model_input_t_.iloc[:,0:6]
    Target_data = model_input_t_.iloc[:,6]
    # Target_data = pd.DataFrame(model_input_, columns=['variable7'])
    # # Z-score 
    # Selected = model_input  # pandas DataFrame
    # Selected_z = Selected

    # z_score = pd.DataFrame(np.abs(stats.zscore(Selected_z)))
    # z_score.index = Selected_z.index
    # z_score.columns = Selected_z.columns

    # Z_score_threshold = 4   # (with Z-score < 4 (NIST standard is '4')
    # Selected_zscore = Selected_z[(z_score < Z_score_threshold).all(axis=1)]

    # [a,b] = np.shape(Selected)
    # [a1,b1] = np.shape(Selected_zscore)

    # removed_zscore = a-a1
    
    # print("Z-score outlier removal process complete... \x0A▼\x0A")
    # print(removed_zscore,"samples are removed\x0A")

    #Simple Moving average process
    pp_data = model_input

    SMA_level = 8 # (8분씩 나눠서 이동평균)
    SF_factor = 1

    [row, col] = np.shape(pp_data)
    pp_data_SMA = pd.DataFrame(np.zeros(pp_data.shape))
    pp_data_SMA.index = pp_data.index
    pp_data_SMA.columns = pp_data.columns

    for i in range(0,col):
        pp_data_SMA.iloc[:,i] = pp_data.iloc[:,i].rolling(window = SMA_level, min_periods = 1, center = True).mean()

    print("Simple moving average process complete...\x0A▼\x0A")

    # Data normalization
    Train_norm_filepath = 'info/'
    Train_Nor = 'Norm_Bound_train.csv'
    pp_data_SMA_Train = pd.read_csv(os.path.join(Train_norm_filepath,Train_Nor), header=None)
    pp_data_SMA_Train = np.array(pp_data_SMA_Train)

    pp_data_SMA = np.array(pp_data_SMA)
    pp_data_Norm = np.zeros(pp_data_SMA.shape)

    # MinMax Normalization
    for i in range(0,pp_data_SMA.shape[1]):
        pp_data_Norm[:,[i]] = (pp_data_SMA[:,[i]]-pp_data_SMA_Train[:,[i]].min()) / (pp_data_SMA_Train[:,[i]].max()-pp_data_SMA_Train[:,[i]].min())
        
    # Training model load & virtual sensor testing 
    model2 = keras.models.load_model('model/MLP_model.h5')
    VS_testing = model2.predict(pp_data_Norm)
    VS_testing = float(VS_testing)
    VS_testing = round(VS_testing,2)

    # Testing target data load
    # Testing_target_data_filepath = 'info/'
    # Testing_target = 'testing_target.csv'
    # Testing_target_data = pd.read_csv(os.path.join(Testing_target_data_filepath,Testing_target), header=0, index_col=0, encoding = 'CP949')
    # Testing_target_data = Testing_target_data.iloc[:,0]

    # FD_threshold_index load
    FD_index_filepath = 'info/'
    FD_index = 'FD_threshold_index.csv'
    FD_threshold_index = pd.read_csv(os.path.join(FD_index_filepath,FD_index), header=None)
    FD_threshold_index = FD_threshold_index.iloc[:,0] # 0 : FD_threshold, 1 : Virtual sensing training accuracy
    # FD_threshold_index = pd.DataFrame(FD_threshold_index)
    
    # Fault detection rate calculation

    
    # VS_testing_ = pd.DataFrame(VS_testing)   # 가상센서 값
    # Testing_target_data = pd.DataFrame(Target_data) # 물리센서 값

    VS_testing_residual = abs(VS_testing-Target_data)
    # VS_testing_residual = pd.DataFrame(VS_testing_residual)

    # Fault detection alram service    
    
    if VS_testing_residual[0] > FD_threshold_index[0]:
        
        sensor_state = "Error occurs!!"
        
        def post_message(token, channel, text):
            response = requests.post("https://slack.com/api/chat.postMessage",
            headers={"Authorization": "Bearer "+token},
            data={"channel": channel,"text": text})

        myToken = "xoxb-1983717984807-2004666170148-aulLTLn0C3yqPLscmPsV1MYK"
        post_message(myToken,"#fdd_alarm_service","Physical sensor error occurs!!")
    else :
        sensor_state = "Normal"


    return jsonify({'Virtual sensor value' : VS_testing, 'Sensor state' : sensor_state,
                    'Residual value' : round(VS_testing_residual[0],2),
                    'Threshold value' : round(FD_threshold_index[0],2)})

if __name__ == '__main__':
    app.run(debug=True)