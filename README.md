# prediction


## ver1 

```
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input # Input 추가됨
import sys
import os

# 불필요한 경고 메시지 끄기 (깔끔한 화면을 위해)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

# --- 1. 데이터 준비 ---
print("📂 로또 데이터를 읽어옵니다...")

def read_csv_safe(filename):
    try:
        return pd.read_csv(filename, encoding='utf-8', header=None)
    except UnicodeDecodeError:
        return pd.read_csv(filename, encoding='cp949', header=None)

try:
    df1 = read_csv_safe('당첨(1~600).csv')
    df2 = read_csv_safe('당첨(601~1203).csv')
    
    data1 = df1.iloc[3:]
    data2 = df2.iloc[3:]
    full_df = pd.concat([data2, data1], axis=0)
    full_df = full_df[[1, 13, 14, 15, 16, 17, 18]]
    full_df.columns = ['Round', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']
    full_df = full_df.apply(pd.to_numeric, errors='coerce').dropna()
    full_df = full_df.sort_values('Round').reset_index(drop=True)
    
    numbers = full_df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values
    scaled_numbers = numbers / 45.0
    
    window_size = 5
    
    def create_dataset(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i : i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)
        
    X, y = create_dataset(scaled_numbers, window_size)
    last_window = scaled_numbers[-window_size:]
    last_window = last_window.reshape((1, window_size, 6))

    print(f"✅ 데이터 읽기 성공! 총 {len(full_df)}회차 데이터를 확보했습니다.")

except Exception as e:
    print(f"\n❌ [오류] 데이터를 읽을 수 없습니다: {e}")
    sys.exit()

# --- 2. 5명의 AI 학습 시작 ---
print(f"\n🚀 5개의 서로 다른 AI 모델 학습을 시작합니다... (잠시만 기다려주세요)")
print("=" * 60)

labels = ['A', 'B', 'C', 'D', 'E']

for i in range(5):
    print(f"\n🤖 [AI 모델 {labels[i]} 학습 중...]")
    
    # 여기서 경고를 없애기 위해 'Input' 층을 따로 만들었습니다.
    model = Sequential([
        Input(shape=(window_size, 6)), # 최신 방식
        LSTM(64, activation='relu'),
        Dense(6)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # 학습 (verbose=0으로 지저분한 로그 숨김)
    model.fit(X, y, epochs=100, batch_size=16, verbose=0)
    
    prediction = model.predict(last_window, verbose=0)
    
    pred_nums = prediction * 45.0
    pred_nums = np.round(pred_nums).flatten().astype(int)
    pred_nums = np.clip(pred_nums, 1, 45)
    
    unique_nums = np.unique(pred_nums)
    while len(unique_nums) < 6:
        new_num = np.random.randint(1, 46)
        if new_num not in unique_nums:
            unique_nums = np.append(unique_nums, new_num)
    
    final_nums = np.sort(unique_nums)
    print(f"👉 Game {labels[i]} 추천 번호: {final_nums}")

print("\n" + "=" * 60)
input("✅ 모든 예측이 끝났습니다. 종료하려면 엔터 키를 누르세요...")
```


## ver2

### 양방향(Bidirectional) LSTM 사용:
기존: 과거 $\rightarrow$ 미래 순서로만 공부했습니다.  
업그레이드: 문맥을 더 잘 파악하기 위해 (과거 $\rightarrow$ 미래)와 (미래 $\rightarrow$ 과거) 양쪽 방향으로 데이터를 훑어보게 만듭니다. (마치 영어 독해를 할 때 앞뒤 문맥을 다 보는 것과 같습니다.)

### 층(Layer) 더 쌓기 (Deep Learning):
기존: 뇌세포 층이 1개였습니다.  
업그레이드: LSTM 층을 2~3개로 겹쳐서 쌓습니다. 1층은 단순한 패턴, 2층은 복잡한 패턴을 분석하도록 **"깊은 사고"**를 하게 만듭니다.

### 학습 횟수와 뇌세포 늘리기:
기존 64개였던 뉴런(Neuron)을 128개 또는 256개로 늘리고, 학습 반복 횟수(Epoch)도 늘려서 더 집요하게 패턴을 찾게 합니다.

```
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, Dropout
import sys
import os

# 1. 환경 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

print("📂 [Pro버전] 변경된 데이터 구조에 맞춰 로딩 중입니다...")

# 2. 데이터 읽기
def read_csv_safe(filename):
    try:
        return pd.read_csv(filename, encoding='utf-8', header=None)
    except UnicodeDecodeError:
        return pd.read_csv(filename, encoding='cp949', header=None)

try:
    df1 = read_csv_safe('당첨(1~600).csv')
    df2 = read_csv_safe('당첨(601~1203).csv')
    
    # 3줄 헤더 건너뛰기 (이건 구조가 유지되었으므로 그대로 둡니다)
    data1 = df1.iloc[3:]
    data2 = df2.iloc[3:]
    
    full_df = pd.concat([data2, data1], axis=0)
    
    # 🔥 [핵심 수정] 변경된 파일 내용에 맞게 열 번호 수정 🔥
    # 기존: [1, 13, 14, 15, 16, 17, 18]
    # 변경: [1, 2, 3, 4, 5, 6, 7] (빈 칸 없이 바로 옆에 붙어 있음)
    full_df = full_df[[1, 2, 3, 4, 5, 6, 7]]
    
    full_df.columns = ['Round', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']
    full_df = full_df.apply(pd.to_numeric, errors='coerce').dropna()
    full_df = full_df.sort_values('Round').reset_index(drop=True)
    
    # 정규화
    numbers = full_df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values
    scaled_numbers = numbers / 45.0
    
    window_size = 10 
    
    def create_dataset(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i : i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)
        
    X, y = create_dataset(scaled_numbers, window_size)
    last_window = scaled_numbers[-window_size:]
    last_window = last_window.reshape((1, window_size, 6))

    print(f"✅ 데이터 로딩 성공! 총 {len(full_df)}회차 (열 구조 자동 보정 완료)")
    print(f"✅ 분석 깊이: 과거 {window_size}회차 데이터를 기반으로 예측")

except Exception as e:
    print(f"\n❌ 데이터 구조 오류: {e}")
    print("👉 엑셀 내용이 바뀌면서 열 번호가 달라진 것 같습니다. 파일 내용을 확인해주세요.")
    sys.exit()

# 3. 고성능 AI 모델 설계 (검사 방법 변경 X)
print(f"\n🚀 [Deep Learning] 고성능 예측 모델 가동 시작...")
print("=" * 60)

labels = ['A', 'B', 'C', 'D', 'E']

for i in range(5):
    print(f"\n🧠 [AI 모델 {labels[i]} 심층 학습 중...]")
    
    model = Sequential([
        Input(shape=(window_size, 6)),
        # 양방향 LSTM
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.2),
        # 심층 LSTM
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        # 결과 출력
        Dense(64, activation='relu'),
        Dense(6)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # 학습 (150회 반복)
    model.fit(X, y, epochs=150, batch_size=16, verbose=0)
    
    # 예측
    prediction = model.predict(last_window, verbose=0)
    
    pred_nums = prediction * 45.0
    pred_nums = np.round(pred_nums).flatten().astype(int)
    pred_nums = np.clip(pred_nums, 1, 45)
    
    unique_nums = np.unique(pred_nums)
    while len(unique_nums) < 6:
        new_num = np.random.randint(1, 46)
        if new_num not in unique_nums:
            unique_nums = np.append(unique_nums, new_num)
    
    final_nums = np.sort(unique_nums)
    print(f"👉 Game {labels[i]} (Deep) 추천: {final_nums}")

print("\n" + "=" * 60)
input("✅ 고성능 예측 완료. 종료하려면 엔터 키를 누르세요...")
```
- 파일 모습도 변경이 되었음 


## ver3
Window Size를 260(약 5년치)를 보고 예측
```
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, Dropout
import sys
import os

# 1. 환경 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

print("📂 [Ultra Long-Term] 과거 260회(약 5년) 데이터를 통째로 분석합니다...")

# 2. 데이터 읽기
def read_csv_safe(filename):
    try:
        return pd.read_csv(filename, encoding='utf-8', header=None)
    except UnicodeDecodeError:
        return pd.read_csv(filename, encoding='cp949', header=None)

try:
    df1 = read_csv_safe('당첨(1~600).csv')
    df2 = read_csv_safe('당첨(601~1203).csv')
    
    # 헤더 3줄 제거
    data1 = df1.iloc[3:]
    data2 = df2.iloc[3:]
    
    full_df = pd.concat([data2, data1], axis=0)
    
    # 수정된 파일 구조 반영 (1~7열)
    full_df = full_df[[1, 2, 3, 4, 5, 6, 7]]
    full_df.columns = ['Round', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']
    full_df = full_df.apply(pd.to_numeric, errors='coerce').dropna()
    full_df = full_df.sort_values('Round').reset_index(drop=True)
    
    # 정규화
    numbers = full_df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values
    scaled_numbers = numbers / 45.0
    
    # 🔥 [핵심 변경] Window Size를 260(약 5년)으로 설정 🔥
    # 너무 크면(예: 1000) 학습할 데이터가 부족해지므로, 260 정도가 적당한 '최대치'입니다.
    window_size = 260
    
    def create_dataset(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i : i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)
        
    X, y = create_dataset(scaled_numbers, window_size)
    last_window = scaled_numbers[-window_size:]
    last_window = last_window.reshape((1, window_size, 6))

    print(f"✅ 데이터 준비 완료! 총 {len(full_df)}회차")
    print(f"✅ 분석 범위: 한 번에 과거 {window_size}주(약 5년)의 흐름을 봅니다.")
    print(f"✅ 학습 가능 예제 수: {len(X)}개 (충분합니다!)")

except Exception as e:
    print(f"\n❌ 오류: {e}")
    sys.exit()

# 3. 모델 학습 및 예측
print(f"\n🚀 [Super Long-Term] 5년치 패턴 정밀 분석 시작...")
print("=" * 60)

labels = ['A', 'B', 'C', 'D', 'E']

for i in range(5):
    print(f"\n🧠 [AI 모델 {labels[i]} 학습 중... (시간이 좀 걸립니다)]")
    
    model = Sequential([
        Input(shape=(window_size, 6)),
        # 5년치 긴 데이터를 까먹지 않게 LSTM 뉴런을 256개로 대폭 늘림
        Bidirectional(LSTM(256, return_sequences=True)),
        Dropout(0.3),
        
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        Dense(6)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # 데이터가 길어서 학습 횟수(epochs)를 300회로 설정
    model.fit(X, y, epochs=300, batch_size=64, verbose=0)
    
    prediction = model.predict(last_window, verbose=0)
    
    pred_nums = prediction * 45.0
    pred_nums = np.round(pred_nums).flatten().astype(int)
    pred_nums = np.clip(pred_nums, 1, 45)
    
    unique_nums = np.unique(pred_nums)
    while len(unique_nums) < 6:
        new_num = np.random.randint(1, 46)
        if new_num not in unique_nums:
            unique_nums = np.append(unique_nums, new_num)
    
    final_nums = np.sort(unique_nums)
    print(f"👉 Game {labels[i]} (5년 분석) 추천: {final_nums}")

print("\n" + "=" * 60)
input("✅ 예측 완료. 종료하려면 엔터 키를 누르세요...")
```

## ver4 
### 전략 변경: "숫자 계산" $\rightarrow$ "확률 선택" (Classification) 
이제부터는 AI에게 숫자를 계산하라고 하지 않고, "1번부터 45번 공 중에서, 나올 확률이 가장 높은 공 6개를 골라봐!"  
기존 방식 (회귀): "다음 숫자는 23.4일 거야" $\rightarrow$ 23 (애매함)  
새로운 방식 (분류): "1번 공이 나올 확률 90%, 2번 공은 10%... 그러니까 1번 추천!" (더 명확함)  
이 방식은 데이터 과학에서 **원-핫 인코딩(One-Hot Encoding)**이라고 부르는 고급 기법을 사용합니다. 
```
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, Dropout, BatchNormalization
import sys
import os

# 1. 환경 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

print("📂 [Final Ver] 확률 기반(Classification) 정밀 분석 모드를 시작합니다...")

# 2. 데이터 읽기
def read_csv_safe(filename):
    try:
        return pd.read_csv(filename, encoding='utf-8', header=None)
    except UnicodeDecodeError:
        return pd.read_csv(filename, encoding='cp949', header=None)

try:
    df1 = read_csv_safe('당첨(1~600).csv')
    df2 = read_csv_safe('당첨(601~1203).csv')
    
    # 헤더 3줄 제거 및 통합
    data1 = df1.iloc[3:]
    data2 = df2.iloc[3:]
    full_df = pd.concat([data2, data1], axis=0)
    
    # 열 구조 정리
    full_df = full_df[[1, 2, 3, 4, 5, 6, 7]]
    full_df.columns = ['Round', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']
    full_df = full_df.apply(pd.to_numeric, errors='coerce').dropna()
    full_df = full_df.sort_values('Round').reset_index(drop=True)
    
    # --- 🔥 여기가 완전히 바뀐 부분입니다 (데이터 가공) 🔥 ---
    # 번호를 그대로 쓰는 게 아니라, 45개의 구멍(One-Hot)을 만듭니다.
    # 예: 당첨번호가 1, 3이면 -> [1, 0, 1, 0, 0, ...] 이런 식의 0과 1로 된 바코드를 만듭니다.
    
    # 데이터 전체를 0~1로 정규화 (입력용)
    numbers = full_df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values
    scaled_numbers = numbers / 45.0
    
    # 결과값(Y)을 위한 원-핫 인코딩 함수
    def numbers_to_onehot(rows):
        # 46개짜리 빈 배열(0번 인덱스는 안 씀)
        onehot = np.zeros((len(rows), 46))
        for i, row in enumerate(rows):
            for num in row:
                onehot[i, int(num)] = 1 # 해당 번호 자리에 1 표시
        return onehot[:, 1:] # 0번 인덱스 제외하고 1~45번만 반환

    # 윈도우 설정 (확률 모델은 너무 길면 오히려 헷갈려해서 50주 정도가 적당함)
    window_size = 50
    
    def create_dataset(raw_data, window_size):
        X, y = [], []
        # raw_data는 입력용 정규화 데이터
        # 실제 번호는 one-hot 타겟용으로 다시 가져옴
        real_numbers = full_df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values
        
        for i in range(len(raw_data) - window_size):
            X.append(raw_data[i : i + window_size])
            # y값은 "다음 회차 번호들의 바코드(One-Hot)"가 됨
            y.append(real_numbers[i + window_size])
            
        return np.array(X), np.array(y)
        
    X, y_indices = create_dataset(scaled_numbers, window_size)
    # y를 원-핫 인코딩으로 변환 (확률 계산용 정답지)
    y = numbers_to_onehot(y_indices) 
    
    last_window = scaled_numbers[-window_size:]
    last_window = last_window.reshape((1, window_size, 6))

    print(f"✅ 데이터 변환 완료: '숫자 예측'이 아닌 '확률 분석' 형태로 변경되었습니다.")

except Exception as e:
    print(f"\n❌ 오류: {e}")
    sys.exit()

# 3. 확률 예측 모델 설계
print(f"\n🚀 [Probability Model] 1~45번 공 각각의 출현 확률을 계산합니다...")
print("=" * 60)

labels = ['A', 'B', 'C', 'D', 'E']

for i in range(5):
    print(f"\n🧠 [AI 모델 {labels[i]} 학습 중...]")
    
    model = Sequential([
        Input(shape=(window_size, 6)),
        
        # 패턴 분석
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        BatchNormalization(), # 학습 안정화 기술 추가
        
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        BatchNormalization(),
        
        Dense(128, activation='relu'),
        
        # 🔥 출력층 변경: 1개의 숫자가 아니라 45개의 확률을 뱉어냄 🔥
        # sigmoid: 각 번호마다 "나올 확률"을 0~100%로 독립적으로 계산
        Dense(45, activation='sigmoid') 
    ])
    
    # 손실 함수 변경: binary_crossentropy (확률 맞추기 전용 채점 방식)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 학습
    model.fit(X, y, epochs=150, batch_size=32, verbose=0)
    
    # 예측 (45개의 확률값이 나옴)
    prob_prediction = model.predict(last_window, verbose=0)[0]
    
    # 확률이 높은 순서대로 6개의 번호 인덱스(위치)를 찾음
    # argsort는 작은 순서대로 정렬하므로, 뒤에서부터 6개를 뽑음 (-6:)
    top_6_indices = prob_prediction.argsort()[-6:]
    
    # 인덱스는 0부터 시작하므로 +1을 해줘야 실제 로또 번호(1~45)가 됨
    final_nums = np.sort(top_6_indices + 1)
    
    # 확률값도 같이 보여주기 (얼마나 확신하는지)
    confidence = prob_prediction[top_6_indices].mean() * 100
    
    print(f"👉 Game {labels[i]} 추천: {final_nums} (AI 확신도: {confidence:.1f}%)")

print("\n" + "=" * 60)
input("✅ 확률 분석 완료. 종료하려면 엔터 키를 누르세요...")
```






