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

### 학습 횟수와 뇌세포 늘리기:기존 64개였던 뉴런(Neuron)을 128개 또는 256개로 늘리고, 학습 반복 횟수(Epoch)도 늘려서 더 집요하게 패턴을 찾게 합니다.


















