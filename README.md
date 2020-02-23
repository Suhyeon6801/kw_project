# kw_project

[팀원] 김수현, 정혜인, 변혜령, 이민아

## 프로젝트 개요 및 작품 설명국
- 국가/국제적 사건에 대한 트위터 데이터와 과거의 환율데이터를 이용하여 환율을 분석한후, 현재 주목받고 있는 딥러닝 기술을 이용하여 7일, 15일, 30일 기간별로 환율을 예측하는 프로그램  

- 환율예측에 두 가지 RNN Model을 사용함. 
1. many to one : 환율 하루씩 예측
2. many to many : 환율은 n일씩 예측 (정확도 ↓)  


## 적용 기술 및 라이브러리
### Machine Learning
 * Tensorflow : Machine Learning을 위한 오픈 소스 -> data를 이용해 Machine Learning 진행
 * Numpy : 데이터를 저장하고 처리하는 다양한 함수 제공 -> data set 처리하는데 사용
 * Pandas : 데이터 분석을 위한 여러가지 라이브러리 제공 -> data-frame을 이용하여 data 가공
### Crawling
 * Beautifulsoup4    
 : html코드를 python이 이해하는 객체구조로 parsing하여 원하는 data를 추출하는데 사용하는 라이브러리   
    -> twitter의 data를   crawling하는데 사용함
### UI
 * Django : 오픈 소스 웹 애플리케이션 프레임워크-> 결과물을 web site에 제공 


### Front-End & Back-End

<img src="https://user-images.githubusercontent.com/28249931/72711141-ce9c0800-3bab-11ea-8cd1-4c551f6a81c0.png" width="500"></img>


## 결과 화면
1. 실시간 환율 ( 한국수출입은행 api사용)

<img src="https://user-images.githubusercontent.com/28249931/72710635-e1faa380-3baa-11ea-89e3-87269d01a0a4.png" width="700"></img>

2. 예측 화면

<img src="https://user-images.githubusercontent.com/28249931/72710777-24bc7b80-3bab-11ea-8a1f-0845b6beaed1.png" width="700"></img>
<img src="https://user-images.githubusercontent.com/28249931/72710782-28500280-3bab-11ea-8698-a9e5341067ba.png" width="700"></img>
<img src="https://user-images.githubusercontent.com/28249931/72710767-1ff7c780-3bab-11ea-8e7b-952e75ae0e70.png" width="700"></img>

3. 정확도 검증

<img src="https://user-images.githubusercontent.com/28249931/72711050-a90efe80-3bab-11ea-9537-0ba238596498.png" width="700"></img>


## 결론
- 국제적 사건에 대한 SNS 데이터와 환율 데이터 간의 상관관계를 기계학습을 통해 학습시키고, 미래의 환율을 예측할 수 있다.
- 실제 환율과 예측 환율에 대한 차이를 한눈에 확인할 수 있다. (프로그램의 신뢰도 확인)
