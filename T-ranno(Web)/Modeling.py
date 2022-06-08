import io                                                         # 파일 경로 접근시 필요한 python 내장 라이브러리                                                               # 강력한 이미지처리와 그래픽 기능 제공 오픈소스 라이브러리
import numpy as np
import pandas as pd
import time
from flask import Flask, render_template                                        # python web framework 
import tensorflow as tf
from flask import Flask, make_response, request                                     # 웹 요청 관련 모듈
from flask import render_template, redirect, url_for          # flask에서 필요한 모듈
from flask import jsonify                                        # import JSON을 해도되지만 여기서는 flask 내부에서 지원하는 jsonify를 사용
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import Sequence                                # 이 모듈이 없으면 사용자가 만든 generator에서 'shape'가 없다고 하는 에러가 발생할 수 있음
from matplotlib import rc
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
import csv
from scipy import stats
import sklearn as sk
import scipy.stats
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error 
from math import sqrt
from datetime import datetime
         
app = Flask(__name__)

@app.route('/')
def form():
    return render_template("Modeling.html")

@app.route('/fileUpload', methods = ['GET', 'POST'])
def method():    
    if request.method == 'GET':
        num = request.args("num")
        return "{}".format(num)

    else:
        filepath = request.files['file']
        data = pd.read_csv(filepath, index_col = 0, encoding = 'CP949', header = 0)
        data = data.iloc[:,0:]

    # Z-score 
    Selected = data  # pandas DataFrame
    Selected_z = Selected

    z_score = pd.DataFrame(np.abs(stats.zscore(Selected_z)))
    z_score.index = Selected_z.index
    z_score.columns = Selected_z.columns

    Z_score_threshold = 4   # (with Z-score < 4 (NIST standard is '4')
    Selected_zscore = Selected_z[(z_score < Z_score_threshold).all(axis=1)]

    [a,b] = np.shape(Selected)
    [a1,b1] = np.shape(Selected_zscore)

    removed_zscore = a-a1
    
    print("Z-score outlier removal process complete... \x0A▼\x0A")
    print(removed_zscore,"samples are removed\x0A")

    # Simple Moving average process
    pp_data = Selected_zscore

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
    pp_data_SMA = np.array(pp_data_SMA)

    pp_data_Norm = np.zeros(pp_data_SMA.shape)

    # MinMax Normalization
    for i in range(0,pp_data_SMA.shape[1]):
        pp_data_Norm[:,[i]] = (pp_data_SMA[:,[i]]-pp_data_SMA[:,[i]].min()) / (pp_data_SMA[:,[i]].max()-pp_data_SMA[:,[i]].min())

    pp_data_Norm = pd.DataFrame(pp_data_Norm)
    
    # Save normalize boundary
    norm_bound = np.zeros((col,2))

    for i in range(0,col):
        norm_bound[[i],0] = [[pp_data_SMA[:,i].min()-(pp_data_SMA[:,i].std())*SF_factor]] # Lower bound..(S.F 0.1 of mean)
        norm_bound[[i],1] = [[pp_data_SMA[:,i].max()+(pp_data_SMA[:,i].std())*SF_factor]] # Upper bound..(S.F 0.1 of mean)

    norm_bound = norm_bound.T

    norm_bound=pd.DataFrame(norm_bound)
    norm_bound.to_csv("info/Norm_Bound_train.csv" # filepath input
                    ,header = False, index = False)
    
    pp_data_SMA = pd.DataFrame(pp_data_SMA)

    print("Train data normalized complete...\x0A▼\x0A")

    # 사용할 데이터만 선택
    input_no = [0,1,2,3,4,5] 
    output_no = [6]

    # model_input = np.array(data.iloc[:,input_no])
    # model_output = np.array(data.iloc[:,output_no])
    model_input = np.array(pp_data_Norm.iloc[:,input_no])
    model_output = np.array(pp_data_SMA.iloc[:,output_no])


    #모델 구조 설정
    model = Sequential()
    model.add(Dense(32, activation = 'relu', input_dim=len(input_no)))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(1, activation='linear'))

    #모델 컴파일링
    model.compile(loss='mean_squared_error',optimizer = "adam")
    early_stoppting = EarlyStopping()

    history = model.fit(model_input, model_output, batch_size=32, epochs=50,
                        verbose=1, validation_split=0.15, shuffle=True, callbacks = [EarlyStopping(monitor="val_loss",
                                                                patience = 5, mode="auto",
                                                                restore_best_weights =True)])

    #학습된 모델로 train data estimation
    Output_pred = model.predict(model_input)

    #FD(Fault Detection) threshold index save
    Output_pred = np.array(Output_pred)  
    FD_index_residual = abs(model_output - Output_pred) # Physical sensor - VS training output
    FD_index_residual = pd.DataFrame(FD_index_residual)

    n=0 # initial counting value

    FD_threshold = 0.65 # FD threshold standard

    for i in range(0,len(FD_index_residual)):
        if FD_index_residual.iloc[i,:][0] > FD_threshold:
            n+=1        

    Training_data_set = len(Output_pred)

    FD_index = n/Training_data_set * 100


    print(100-round(FD_index,2)) # FD index (VS training accuracy)

    FD_threshold_index = [FD_threshold, 100-round(FD_index,2)]

    FD_threshold_index = pd.DataFrame(FD_threshold_index).T

    FD_threshold_index.to_csv("info/FD_threshold_index.csv", header = False, index = False)

    #모델 저장하기
    model.save("model/MLP_model.h5")
    now = time.strftime('%y-%m-%d %H:%M:%S')


    #RMSE formulate
    mse = mean_squared_error(model_output, Output_pred)
    rmse = sqrt(mse)
    rmse = round(rmse,3)
    #Virtual sensor training result plot    
    Output_fig = plt.figure(dpi=300)

    plt.plot(model_output,label='Physical sensor',color='#1C00FF',linewidth = 1)
    plt.plot(Output_pred,label='Virtual sensor',color='#FF0000', linewidth = 1)

    rc('font', family='serif')
    rc('font', serif='Times New Roman')
    # plt.rcParams['font.family'] = 'times new roman'
    plt.xlabel('Time [minute]',fontsize=14)
    plt.ylabel('Temperature [℃]',fontsize=14)
    plt.title('Virtual sensor training result',fontsize=14)
    plt.grid(linestyle='--', color='#BAC0D2')
    plt.legend(loc='upper right')
    plt.rcParams["figure.figsize"] = (20,15)
    plt.ylim(38,46)
    plt.tick_params(axis='x', direction='in', length=3, pad=6, labelsize=12, labelcolor='black', top=True)
    plt.tick_params(axis='y', direction='in', length=3, pad=6, labelsize=12, labelcolor='black', top=True)
    plt.savefig('static/css/modeling_result.png')
    
    return render_template('Modeling_result.html',RMSE=rmse, Now_time=now)
    
if __name__ == '__main__':
    app.run(port=5001,debug=True)