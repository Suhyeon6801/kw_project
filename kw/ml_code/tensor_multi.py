"""
단순 환율예측 코드
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import json
import csv
import matplotlib.pyplot as plt

tf.set_random_seed(777)
input_data_column_cnt = 5  # 입력데이터의 컬럼 개수(Variable 개수)
# 'election','trump','Now', 'Open', 'High', 'Low', 'Volume'
output_data_column_cnt = 1  # 결과데이터의 컬럼 개수

seq_length = 28 
rnn_cell_hidden_dim = 20
forget_bias = 1.0
num_stacked_layers = 1
keep_prob = 0.7

epoch_num = 10000
learning_rate = 0.01  # 학습률

# Standardization
def data_standardization(x):
    x_np = np.asarray(x)
    return (x_np - x_np.mean()) / x_np.std()

# Min-Max scaling
def min_max_scaling(x):
    x_np = np.asarray(x)
    #print("min %s" % x_np.min(()))
    #print("min %s" % x_np.max(()))
    #print((x_np.max() - x_np.min() + 1e-7))
    #print((x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7))
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7)  # 1e-7은 0으로 나누는 오류 예방차원

def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()


def data_load(dataframe, index):
    raw_dataframe = dataframe
    exchange_info = raw_dataframe.values[1:].astype(np.float64)
    price = exchange_info[:, :-1]
    #print(price)
    norm_price = min_max_scaling(price)

    # 거래량형태 데이터를 정규화한다
    # [ 'election','trump','Now', 'Open', 'High', 'Low', 'Volume']에서 마지막 'Volume'만 취함
    volume = exchange_info[:, -1:]
    norm_volume = min_max_scaling(volume)
    x = np.concatenate((norm_price, norm_volume), axis=1)

    y = x[:, [index]]

    dataX = []  # 입력으로 사용될 Sequence Data
    dataY = []  # 출력(타켓)으로 사용
    data_day = []  # 날짜

    for i in range(0, len(y) - seq_length):
        _x = x[i: i + seq_length]
        _y = y[i + seq_length]
        if i is 0:
            print("#")
        dataX.append(_x)
        dataY.append(_y)

    # 학습용/테스트용 데이터 생성
    # 전체 70%를 학습용 데이터로 사용
    train_size = int(len(dataY) * 0.7)
    # 나머지(30%)를 테스트용 데이터로 사용
    test_size = len(dataY) - train_size

    trainX = np.array(dataX[0:train_size])
    trainY = np.array(dataY[0:train_size])

    testX = np.array(dataX[train_size:len(dataX)])
    testY = np.array(dataY[train_size:len(dataY)])

    X = tf.placeholder(tf.float32, [None, seq_length, input_data_column_cnt])

    Y = tf.placeholder(tf.float32, [None, 1])

    targets = tf.placeholder(tf.float32, [None, 1])
    predictions = tf.placeholder(tf.float32, [None, 1])

    stackedRNNs = [lstm_cell() for _ in range(num_stacked_layers)]
    multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs,
                                              state_is_tuple=True) if num_stacked_layers > 1 else lstm_cell()

    hypothesis, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)

    hypothesis = tf.contrib.layers.fully_connected(hypothesis[:, -1], output_data_column_cnt, activation_fn=tf.nn.relu)

    loss = tf.reduce_sum(tf.square(hypothesis - Y))
    optimizer = tf.train.AdamOptimizer(learning_rate)

    train = optimizer.minimize(loss)

    rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)))

    train_error_summary = []  # 학습용 데이터의 오류를 중간 중간 기록한다
    test_error_summary = []  # 테스트용 데이터의 오류를 중간 중간 기록한다
    test_predict = ''  # 테스트용데이터로 예측한 결과

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    start_time = datetime.datetime.now()
    print('학습을 시작합니다...')
    for epoch in range(epoch_num):
        _, _loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
        if ((epoch + 1) % 100 == 0) or (epoch == epoch_num - 1):  # 100번째마다 또는 마지막 epoch인 경우
            train_predict = sess.run(hypothesis, feed_dict={X: trainX})
            train_error = sess.run(rmse, feed_dict={targets: trainY, predictions: train_predict})
            train_error_summary.append(train_error)

            test_predict = sess.run(hypothesis, feed_dict={X: testX})
            test_error = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
            test_error_summary.append(test_error)


            print("epoch: {}, train_error(A): {}, test_error(B): {}, B-A: {}".format(epoch + 1, train_error, test_error,
                                                                                     test_error - train_error))

    end_time = datetime.datetime.now()

    ad_value = 0.03

    for i in range(len(testY)):
        a = testY[i]
        b = test_predict[i]
        dif = a - b
        if abs(dif) > ad_value:
            if dif < 0:
                test_predict[i] = a + ad_value
            elif dif > 0:
                test_predict[i] = a - ad_value

    recent_data = np.array([x[len(x) - seq_length:]])
    test_predict = sess.run(hypothesis, feed_dict={X: recent_data})

    print("test_predict", test_predict[0])
    test_predict = reverse_min_max_scaling(price, test_predict)  # 금액데이터 역정규화한다
    print("Tomorrow's exchange price", test_predict[0])  # 예측한 환율를 출력한다

    return test_predict


def lstm_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_cell_hidden_dim,reuse=tf.AUTO_REUSE,
                                        forget_bias=forget_bias, state_is_tuple=True, activation=tf.sigmoid)

    if keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell

def get_new_Data(dataframe):
    predict_col = dataframe.shape[1] - 2
    #print(predict_col)
    append_cell = []

    for j in range(predict_col+1):
        print("target j : "+str(j))
        predict = data_load(dataframe, j)
        predict = predict[0]
        append_cell.append(predict[0])
        tf.reset_default_graph()

    yester_now = dataframe.iloc[dataframe.shape[0]-1, 0]
    volume = (float(append_cell[1]) - float(yester_now)) / float(yester_now)
    append_cell.append(volume)
    new_data = pd.DataFrame(data=[append_cell], columns=['Now', 'Open', 'High', 'Low', 'Volume'])
    print(new_data)
    print (append_cell)

    dataframe =dataframe.append(new_data, sort=False)

    return dataframe

if __name__=="__main__":
    n = 30
    # 데이터를 로딩한다.
    stock_file_name = 'INPUT_FIN.csv'  # 환율 데이터 파일
    encoding = 'utf-8-sig'  # 문자 인코딩
    names = ['Date', 'Now', 'Open', 'High', 'Low', 'Volume']
    load_data = pd.read_csv(stock_file_name, names=names, encoding=encoding)  # 판다스이용 csv파일 로딩
    print(load_data.shape[1])
    load_data.info()  # 데이터 정보 출력
    raw_dataframe = load_data
    del raw_dataframe['Date']

    for a in range(n):
        print("the day : " + str(a))
        load_data = get_new_Data(load_data)

    load_data.to_csv('FINAL.csv', index=False, header=False)

    load_data.to_json('FINAL.json', orient='table')

    del load_data
