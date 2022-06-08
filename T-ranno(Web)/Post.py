
import requests, json
import pandas as pd
import matplotlib.pyplot as plt



test_raw_data = pd.read_csv("info/test.csv",index_col=0)
input_data = test_raw_data.iloc[:,0:6]
Target_data = test_raw_data.iloc[:,6]

headers = {'Content-Type': "application/json"}
URL ='http://127.0.0.1:5000/testing'  ## 발급받은 API 키값
x=0
x_init = [0]
y_phy_init = [40]
y_vir_init = [40]
resi_val_init = [0]
threshold_val_init = [0.65]
for i in range(0,len(test_raw_data)):
    list_data = input_data.iloc[i,:]
    list_data = list(list_data)

    TG_data = Target_data[i]

    A = list_data[0]
    B = list_data[1]
    C = list_data[2]
    D = list_data[3]
    E = list_data[4]
    F = list_data[5]
    T = TG_data # 물리센서 값
    data = {"Testing input_1": A,"Testing input_2": B,"Testing input_3": C,"Testing input_4": D,"Testing input_5": E,"Testing input_6": F,"Testing input_7": T}
    res = requests.post(URL, headers=headers, data=json.dumps(data))

    # 실시간 그래프 제작
    xml_dic = res.json()
    all_key_list = list(xml_dic.values())
    # print("리스트가 어떻게되나요!",all_key_list)

    # 갱신 data
    x = x + 1
    y_phy = T
    threshold_val = all_key_list[2] # 임계치 값
    resi_val = all_key_list[0] # 잔차 값
    y_vir = all_key_list[3] # 가상센서 값

    x_init.append(x)
    y_phy_init.append(y_phy)
    y_vir_init.append(y_vir)
    resi_val_init.append(resi_val)
    threshold_val_init.append(threshold_val)


    plt.subplot(2, 1, 1)
    plt.plot(y_phy_init,color='navy',label="Physical sensor value")
    plt.plot(y_vir_init,color='darkred',label="Virtual sensor value")
    if x >= 1:
        plt.xlim(x-5, x+30)

    while i ==0:
        plt.legend(loc='upper right', ncol=2)
        plt.ylim(35, 50)
        break

    plt.subplot(2, 1, 2)
    plt.plot(resi_val_init,color='black',label="Residual value")
    plt.plot(threshold_val_init,color='red',label="Threshold value")

    if x >= 1:
        plt.xlim(x-5, x+30)

    while i ==0:
        plt.legend(loc='upper right')
        plt.ylim(0, 2)
        break


    # print(y_phy_init)
    # print(y_vir_init)
    plt.pause(0.05)
    # plt.show()
    print(xml_dic)

plt.show()



