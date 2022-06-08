import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
import os
import numpy as np
from numpy import genfromtxt
from numpy import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import scipy.stats
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from tensorflow.keras.models import load_model
# from tensorflow.python.keras.models import load_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error



form_class = uic.loadUiType("C:/Users/LTS/OneDrive - inu.ac.kr/바탕 화면/새 폴더 (2)/T-RANO TECHNOLOGY.ui")[0]



class Modelingwindow(QDialog):
    
    def __init__(self, parent):
        super(Modelingwindow, self).__init__(parent)
        uic.loadUi("C:/Users/LTS/OneDrive - inu.ac.kr/바탕 화면/새 폴더 (2)/Training_pop-up.ui", self)
        self.show()
        
        self.pushButton_upload.clicked.connect(self.upload)
        self.pushButton_modeling.clicked.connect(self.modeling)

    def upload(self):

        fname = QFileDialog.getOpenFileName(self)
        if fname[0]:
            with open(fname[0], encoding='CP949') as f:

                data = f.read()
            #print("data_path-> {}".format(fname[0]))
            csv_path = "{}".format(fname[0])
            global file_path
            file_path = csv_path.replace("/","//")
            print(file_path)

        self.textEdit_2.append(file_path)

        
    def modeling(self):

        #%% (1): Basic settings
   
        # 자주 사용되는 Python library Import
        #import os
        import numpy as np
        import pandas as pd
        from numpy import genfromtxt
        from numpy import sqrt
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import sklearn as sk
        import scipy.stats
        
        # Data 불러오기 
        dt_train_ = pd.read_csv(file_path,index_col = 0,encoding = 'CP949')
       
        # # 사용할 데이터만 선택
        selected_no = [0,1,2,3,4,5,6]

        dt_train = dt_train_.iloc[:,selected_no]
        
         # (2-1): 이동평균(Simple Moving Average)

        def SMA(df_np_,level):
            import numpy as np
            import pandas as pd
            
            df_ = pd.DataFrame.copy(pd.DataFrame(df_np_))
            df_SMA_ = pd.DataFrame.copy(pd.DataFrame(np.zeros(np.shape(df_np_))))
            for i in range(0,np.shape(df_SMA_)[1]):
                df_SMA_.loc[:,i] = df_.loc[:,i].rolling(window = level, min_periods = 1,center = True).mean()  
            
            df_SMA = np.array(df_SMA_)
            print("Data SMA completed!")
            return df_SMA

        dt_pre2_1 = pd.DataFrame(SMA(np.array(dt_train),10), columns = dt_train.columns)

        # (2-2): 데이터 전처리 - 이상치 제거 (Outlier removal) 

        def z_score_Outlier_rm(df_np,limit):
            import scipy
            from scipy import stats
            z_scores = stats.zscore(df_np)
            abs_z_scores = np.abs(z_scores)
            filtered_entries = (abs_z_scores<limit).all(axis=1)
            new_df = np.copy(df_np[filtered_entries])
            Deleted = len(df_np) - len(new_df)    
            print("Deleted outlier:",Deleted,"tuples",round((100-len(new_df)/len(df_np)*100),2),"%")
            print("Data Outlier removal completed!")
            return(new_df) # new_df: 원본데이터 스케일 new_score: z-score로 계산된 스케일

        dt_pre2_2 = pd.DataFrame(z_score_Outlier_rm(np.array(dt_pre2_1),3), columns = dt_train.columns)

        # (2-3): 데이터 전처리 - 정규화 과정 [0,1] 

        def Normboundary_gen(df_):
            norm_bound = np.zeros((np.shape(df_)[1],2))
            SF_factor = 1
            for i in range(0,np.shape(df_)[1]):
                #각 변수 별 최대 최소 값을 산출해서 행렬로 저장
                norm_bound[[i],0] = [[df_[:,i].min()]]
                norm_bound[[i],1] = [[df_[:,i].max()]]        
            return norm_bound

        def Normalize(df_,bound):
            df_norm = np.zeros(np.shape(df_))
            for i in range(0,np.shape(df_)[1]):
                # [0,1] 범위로 정규화 계산을 진행
                df_norm[:,[i]] = (df_[:,[i]]-bound[[i],0])/ (bound[[i],1]-bound[[i],0])
            print("Data Normalization completed!")    
            return df_norm

        Normalization_boundary = Normboundary_gen(np.array(dt_pre2_2))
        dt_pre2_3 = pd.DataFrame(Normalize(np.array(dt_pre2_2),Normalization_boundary), columns = dt_train.columns)

        # (3-1): 모델 학습에 사용할 입력변수, 출력변수 설정 

        input_no = [0,1,2,3,4,5] 
        output_no = [6]

        model_input = np.array(dt_pre2_3.iloc[:,input_no])
        model_output = np.array(dt_pre2_3.iloc[:,output_no])

        # (3-2): 모델 학습과정

        #모델 학습을 위한 Library 가져오기
        import keras
        from tensorflow.keras.models import Sequential, Model
        from tensorflow.keras.layers import Dense, Activation
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import TensorBoard
        from sklearn.metrics import r2_score
        from sklearn.metrics import mean_squared_error
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        #모델 구조 설정
        model = Sequential()
        model.add(Dense(32, activation = 'relu', input_dim=len(input_no)))
        model.add(Dense(32, activation = 'relu'))
        model.add(Dense(32, activation = 'relu'))
        model.add(Dense(1, activation='linear'))

        #모델 컴파일링
        model.compile(loss='mean_squared_error',optimizer = "adam")

        history = model.fit(model_input, model_output, batch_size=32, epochs=50,
                            verbose=1, validation_split=0.15,shuffle=True)

        #학습된 모델로 train data estimation
        Output_pred = model.predict(model_input)


        #모델 저장하기
        #model.save("C://PyQt Tutorial//5. TAESUNG COMPANY 가상센서//model.h5")
        # Base_dir = os.path.dirname(file_path)
        model.save("C:/Users/LTS/OneDrive - inu.ac.kr/바탕 화면/새 폴더 (2)/your_model.h5")


        # # (4-1): 정규화된 상태인 출력 변수 역정규화하기

        output_norm_bound = Normalization_boundary[output_no,:]
        true_rv_norm = output_norm_bound[0,0] + np.array(model_output)*(output_norm_bound[0,1]-output_norm_bound[0,0])
        ypred_rv_norm = output_norm_bound[0,0] + Output_pred*(output_norm_bound[0,1]-output_norm_bound[0,0])

        # # (4-2): 모델 평가지표 R2 score, RMSE

        R2_model = r2_score(true_rv_norm,ypred_rv_norm)
        RMSE_model = mean_squared_error(true_rv_norm,ypred_rv_norm)

        R2_value = round(R2_model,3)
        #print("RMSE value:",round(RMSE_model,5),"\x0A")
        RMSE_value = round(RMSE_model,5)
       
        self.textEdit_3.append(str(R2_value))
        self.textEdit_4.append(str(RMSE_value))



        # graph 시각화
        
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        self.GraphLayout.addWidget(self.canvas)
       
        ax = self.fig.add_subplot(111)
        ax.plot(np.arange(0,len(model_output),1),true_rv_norm,label = 'True',color = 'magenta')
        ax.plot(np.arange(0,len(Output_pred),1),ypred_rv_norm,label = 'Estimated',color = 'dodgerblue')
       
        ax.set_title("Training Result")
        ax.legend(loc='upper right')
        ax.grid()
        ax.margins(x=0)
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Variable")

        self.canvas.draw()

# class Modeluploadwindow(QDialog):
    
    # def __init__(self, parent):
        # super(Modeluploadwindow, self).__init__(parent)
        # uic.loadUi("C:\\Users\\kk390\\Desktop\\2021-1\\06.창업동아리\\pyQt\\13주차\\파일 업로드.ui", self)
        # self.show()
        
        ## self.pushButton_testing.clicked.connect(self.Sensing)
        # self.pushButton_upload_model.clicked.connect(self.upload)


    # def upload(self):
    
        # fname = QFileDialog.getOpenFileName(self)
        
        # model_path = "{}".format(fname[0])
        # global model_file_path
        # model_file_path = model_path.replace("/","//")

        # elf.textEdit_upload_model.append(model_file_path)

    # def Sensing(self):
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
# from PyQt5.QtCore import QDateTime, Qt
from PyQt5.QtCore import QTimer, QTime
import datetime

class WindowClass(QMainWindow, form_class):
    

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.timeout)
        self.timer.start()
    
    def timeout(self):
        currentDate = datetime.datetime.now()
        self.dateEdit.setDate(currentDate.today())
        # self.timeEdit.setTime(currentTime.time())
        currentTime = QTime.currentTime().toString("hh:mm:ss")
        self.label.setText(str(currentTime))
 

    # app = QApplication(sys.argv)
    # QMainWindow = QMainWindow()
    # QMainWindow.show()
    # sys.exit(app.exec_())

        self.pushButton_Modelingbtn.clicked.connect(self.Modelingbtn)
        self.pushButton_testing.clicked.connect(self.Sensingbtn)
        # self.statusBar().showMessage(self.datetime.toString(Qt.DefaultLocaleShortDate))
        # #self.textEdit_3.append(self.datetime.toString(Qt.DefaultLocaleLongDate))
        # self.textEdit_3.append(self.datetime.toString("yyyy-MM-dd"))
        # self.textEdit_4.append(self.datetime.toString("hh:mm:ss"))
        # # self.action_upload.triggered.connect(self.Modelupload)

    def Modelingbtn(self):
        Modelingwindow(self)

    # def Sensingbtn(self):
        # Modeluploadwindow(self)

    # def Modelupload(self):
        # Modeluploadwindow(self)

    # # (5): 모델 테스트하기
    

    ## 센싱버튼
    def Sensingbtn(self):

        #모델 학습을 위한 Library 가져오기
        import keras
        from tensorflow.keras.models import Sequential, Model
        from tensorflow.keras.layers import Dense, Activation
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import TensorBoard
        from sklearn.metrics import r2_score
        from sklearn.metrics import mean_squared_error
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        
        # 테스팅 인풋
        dt_test_  = pd.read_csv(("C:/Users/LTS/OneDrive - inu.ac.kr/바탕 화면/새 폴더 (2)/test.csv"),index_col = 0,encoding = 'CP949')

        # # 사용할 데이터만 선택
        selected_no = [0,1,2,3,4,5,6]

        dt_test = dt_test_.iloc[:,selected_no]

        input_no = [0,1,2,3,4,5] 
        output_no = [6]


        # (2-1): 이동평균(Simple Moving Average)

        def SMA(df_np_,level):
            import numpy as np
            import pandas as pd
            
            df_ = pd.DataFrame.copy(pd.DataFrame(df_np_))
            df_SMA_ = pd.DataFrame.copy(pd.DataFrame(np.zeros(np.shape(df_np_))))
            for i in range(0,np.shape(df_SMA_)[1]):
                df_SMA_.loc[:,i] = df_.loc[:,i].rolling(window = level, min_periods = 1,center = True).mean()  
            
            df_SMA = np.array(df_SMA_)
            print("Data SMA completed!")
            return df_SMA

        test_pre2_1 = pd.DataFrame(SMA(np.array(dt_test),10), columns = dt_test.columns)


        # #전처리(이상치제거): 테스트에서는 하지 않음

        # #전처리(정규화)
        
        def Normboundary_gen(df_):
            norm_bound = np.zeros((np.shape(df_)[1],2))
            SF_factor = 1
            for i in range(0,np.shape(df_)[1]):
                #각 변수 별 최대 최소 값을 산출해서 행렬로 저장
                norm_bound[[i],0] = [[df_[:,i].min()]]
                norm_bound[[i],1] = [[df_[:,i].max()]]        
            return norm_bound

        def Normalize(df_,bound):
            df_norm = np.zeros(np.shape(df_))
            for i in range(0,np.shape(df_)[1]):
                # [0,1] 범위로 정규화 계산을 진행
                df_norm[:,[i]] = (df_[:,[i]]-bound[[i],0])/ (bound[[i],1]-bound[[i],0])
            print("Data Normalization completed!")    
            return df_norm

        Normalization_boundary = Normboundary_gen(np.array(test_pre2_1))
        test_pre2_2 = pd.DataFrame(Normalize(np.array(test_pre2_1),Normalization_boundary), columns = dt_test.columns)

        # #입출력변수 데이터 선택
        xtest_ = test_pre2_2.iloc[:,input_no]
        ytest_ = test_pre2_2.iloc[:,output_no]

        # #모델로 추론한 변수 산출
        model = load_model("C:/Users/LTS/OneDrive - inu.ac.kr/바탕 화면/새 폴더 (2)/your_model.h5")
        ypred_ = model.predict(xtest_)

        # #정규화된 상태인 출력 변수 역정규화하기
        
        output_norm_bound = Normalization_boundary[output_no,:]
        ytest_rv_norm_ = output_norm_bound[0,0] + np.array(ytest_)*(output_norm_bound[0,1]-output_norm_bound[0,0])
        ypred_rv_norm_ = output_norm_bound[0,0] + ypred_*(output_norm_bound[0,1]-output_norm_bound[0,0])

        # #모델 평가지표 R2 score, RMSE
        R2_model_ = r2_score(ytest_rv_norm_,ypred_rv_norm_)
        RMSE_model_ = mean_squared_error(ytest_rv_norm_,ypred_rv_norm_)

        
        # graph 시각화
        
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        self.GraphLayout_test.addWidget(self.canvas)
       
        ax = self.fig.add_subplot(111)
        ax.plot(np.arange(0,len(ytest_rv_norm_),1),ytest_rv_norm_,label = 'True',color = 'magenta')
        ax.plot(np.arange(0,len(ypred_rv_norm_),1),ypred_rv_norm_,label = 'Estimated',color = 'dodgerblue')
       
        ax.set_title("Virtual Sensing Result")
        ax.legend(loc='upper right')
        ax.grid()
        ax.margins(x=0)
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Variable")

        self.canvas.draw()

        #pass
        


app = QApplication(sys.argv)
mainWindow = WindowClass()
mainWindow.show()
app.exec_()