import json
import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# dateutil is an extension package to the python standard datetime module.
import numpy as np

## 참고자료
# numpy 기초
# http://pythonstudy.xyz/python/article/402-numpy-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0
# http://aikorea.org/cs231n/python-numpy-tutorial/

# matplotlib 기초
# https://datascienceschool.net/view-notebook/d0b1637803754bb083b5722c9f2209d0/

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import scipy.spatial as spatial

# 사이파이. 점들간의 거리계산을 위한 것인듯
import matplotlib.dates as mdates

from pandas import DataFrame as df
import pandas as pd

date_list = []  # 엑스축

usd_list = []  # 와이축1
jpy_list = []  # 와이축2

usd_dictionary = {}  # ?
jpy_dictionary = {}  # ?

date_iterator = (
    datetime.today() - relativedelta(months=3)
).date()  # 지금 시간에서 3달전의 시간을 해당 변수에 저장

while date_iterator != datetime.today().date():  # 같아질때까지
    date_list.append(date_iterator)  # date_list에 넣고
    date_iterator += timedelta(days=1)  # 더하기

# relativedelta
# timedelta

# http://egloos.zum.com/mcchae/v/11203068 참고! (한국어)
# relativedelta를 쓴 이유는
# timedelta는 시분초, day week까지만 가능. month가 지원 X

for date in date_list:  # date랑 data랑 헷갈 ㄴㄴ
    url = (
        "https://www.koreaexim.go.kr/site/program/financial/exchangeJSON?authkey=y3yL7pP10tUmqA7YW8vCr0AXhbqemgKa"
        + "&searchdate="
        + date.strftime("%Y%m%d")
        + "&data=AP01"
    )
    res = requests.get(url)  # date에 관련된 정보에 해당하는 url로 response로 가져오고
    data = json.loads(res.text)  # 받아온 json형태의 데이터를 바꿔줌
    # json.loads는?
    # python의 json decoder.
    # When receiving data from a webserver, the data is always a string.
    # If you have a JSON string, you can parse it by using the json.loads() method.

    # -> the result is a Python dictionary!
    # 근데 리스트도 될 수 있나

    # 이와 반대로 (서버입장에서) Python object를 JSON형태로 바꾸고싶다면
    # json.dumps() 메소드를 사용한다.

    print("받아온 restxt", res.text)
    print("json.loads 된 data", data)  # 딕셔너리들이 들어있는 리스트임을 확인할 수 있다

    if str(data) == "[]":  # data가 비었다면 (아무 정보도 가져오지 못했다면)
        usd_list.append("No data")
        jpy_list.append("No data")

    else:  # 뭐든 정보를 가져왔을 경우
        for item in data:  # 리스트 안에 있는 하나의 딕셔너리를 'item'이라는 이름으로 가져다가
            # 'cur_unit'은 해당 딕셔너리에서 어느나라의 통화인지 알려주는 key의 이름
            if item["cur_unit"] == "USD":  # 딕셔너리의 정보가 미국돈 정보라면
                usd_list.append(
                    float(item["ttb"].replace(",", ""))
                )  # ttb라는 key에 해당하는 정보를 가져오는데 ,가 있다면 공백으로 바꿔주고 float화 시켜줘서 usd_list에 마지막 요소로 추가시켜준다(append).
            if item["cur_unit"] == "JPY(100)":  # 일본돈도 마찬가지
                jpy_list.append(float(item["ttb"].replace(",", "")))

                # 참고 : ttb : Telegraphic Transfer Buying, 전신환매입율
                # -> 은행에서 고객으로부터 외국환을 살때 적용하는 환율(고객입장에서는 팔때환율)
                #        tts : Telegraphic Transfer Selling, 전신환매도율
                # -> 은행에서 고객에게 외국환을 팔때 적용하는 환율 (고객입장에서 살때 환율)
                # 그러니 보통은 tts가 더 높겠죠?


for i in range(0, len(usd_list)):  # usd_list에 iterate 해서 dictionary에 date를 키로하고 저장
    usd_dictionary[date_list[i]] = usd_list[i]


for key in list(usd_dictionary.keys()):  # usd_dictionary의 키(date -> 날짜!) 하나하나를 가져다가 반복문
    if usd_dictionary[key] == "No data":  # 만약 해당 키의 value가 "No data" 라면,
        usd_dictionary.pop(key)  # 해당 키와 value를 dictionary에서 제거(pop)한다.

for i in range(0, len(jpy_list)):
    jpy_dictionary[date_list[i]] = jpy_list[
        i
    ]  # 같은 방식으로 date를 키로 해서 jpy_list를 jpy_dictionary로 바꿔준다

for key in list(jpy_dictionary.keys()):
    if jpy_dictionary[key] == "No data":  #  No data인 날짜(key)랑 value는
        jpy_dictionary.pop(key)  # 빼버린다.


# 본격적으로 그래프를 그리기 위해
# 먼저 usd dictionary의 key값(날짜)을 x축으로,
# value (해당 날짜의 ttb)를 y축으로 리스트화해서 저장
x_usd = list(usd_dictionary.keys())
y_usd = list(usd_dictionary.values())

# jpy도 똑같이
x_jpy = list(jpy_dictionary.keys())
y_jpy = list(jpy_dictionary.values())


# subplots() => Create a figure and a set of subplots.
# https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.subplots.html
fig, ax1 = plt.subplots()

# "r-" 이라는 인자로 그래프의 스타일, 즉 색상을 red로.
# 마찬가지로 jpy는 "g-" 로서 색상을 green으로
ax1.plot(x_usd, y_usd, "r-", label="usd")
ax1.plot(x_jpy, y_jpy, "g-", label="jpy")

ax1.grid()  # 그리드를 사용
ax1.legend()  # 범례 (각 plot label 표시)
plt.show()  # 띄워주는거
# plt.savefig('바탕 화면/foo.png') # 만든 plot을 저장
