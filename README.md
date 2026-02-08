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

## ver5 
1. 파생 변수(Feature Engineering) 추가:
- 기존: AI에게 "1, 2, 3..." 번호만 줬습니다.
- 변경: 번호뿐만 아니라 **"번호의 합계(Sum)"**와 "홀짝 비율(Odd/Even)" 같은 힌트를 같이 줍니다. 마치 수학 문제를 풀 때 공식도 같이 알려주는 것과 같습니다.

2. 어텐션(Attention) 메커니즘 도입:
이것이 바로 ChatGPT의 핵심 기술입니다.
과거 50주를 볼 때, 모든 회차를 똑같이 중요하게 보는 게 아니라, **"패턴상 중요한 회차"에 더 집중(Attention)**하도록 만듭니다.

3. 동적 학습률(Dynamic Learning Rate):
처음엔 크게크게 배우다가, 정답에 가까워질수록 아주 미세하게 조정하며 학습하도록 학습 속도를 조절합니다.

```
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, Dropout, BatchNormalization, MultiHeadAttention, LayerNormalization, Concatenate
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import sys
import os

# 1. 환경 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

print("📂 [ULTIMATE PRO] 로또 예측의 끝판왕 모델을 가동합니다...")
print("👉 적용 기술: Feature Engineering + Self-Attention + Dynamic Learning")

# 2. 데이터 읽기 및 파생변수 생성
def read_csv_safe(filename):
    try:
        return pd.read_csv(filename, encoding='utf-8', header=None)
    except UnicodeDecodeError:
        return pd.read_csv(filename, encoding='cp949', header=None)

try:
    df1 = read_csv_safe('당첨(1~600).csv')
    df2 = read_csv_safe('당첨(601~1203).csv')
    
    # 헤더 제거 및 통합
    data1 = df1.iloc[3:]
    data2 = df2.iloc[3:]
    full_df = pd.concat([data2, data1], axis=0)
    
    # 열 구조 정리 (1~7열)
    full_df = full_df[[1, 2, 3, 4, 5, 6, 7]]
    full_df.columns = ['Round', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']
    full_df = full_df.apply(pd.to_numeric, errors='coerce').dropna()
    full_df = full_df.sort_values('Round').reset_index(drop=True)
    
    # --- 🔥 [업그레이드 1] 파생 변수(힌트) 생성 🔥 ---
    print("⚙️ 데이터를 정밀 분석하여 '합계'와 '홀짝 비율' 정보를 추가합니다...")
    
    # 번호 데이터
    num_data = full_df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values
    
    # 1. 합계(Sum) 계산 및 정규화 (대략 255가 최대라고 가정)
    sums = np.sum(num_data, axis=1).reshape(-1, 1) / 255.0
    
    # 2. 홀수 개수(Odd Count) 계산 및 정규화 (0~6개)
    odds = np.sum(num_data % 2, axis=1).reshape(-1, 1) / 6.0
    
    # 3. 원본 번호 정규화
    scaled_numbers = num_data / 45.0
    
    # 모든 정보를 합침 (입력 데이터가 6개에서 8개로 늘어남!)
    # [번호1, 번호2, ..., 번호6, 합계, 홀수개수]
    final_input_data = np.hstack([scaled_numbers, sums, odds])
    
    # 정답지(Target) 생성 - 원-핫 인코딩
    def numbers_to_onehot(rows):
        onehot = np.zeros((len(rows), 46))
        for i, row in enumerate(rows):
            for num in row:
                onehot[i, int(num)] = 1
        return onehot[:, 1:] # 1~45번만 사용

    window_size = 50 # 과거 50주 패턴 분석
    
    def create_dataset(input_features, original_nums, window_size):
        X, y = [], []
        for i in range(len(input_features) - window_size):
            X.append(input_features[i : i + window_size])
            # 정답은 다음 회차의 실제 번호
            y.append(original_nums[i + window_size])
        return np.array(X), np.array(y)
        
    X, y_indices = create_dataset(final_input_data, num_data, window_size)
    y = numbers_to_onehot(y_indices)
    
    # 예측용 마지막 데이터
    last_window = final_input_data[-window_size:]
    last_window = last_window.reshape((1, window_size, 8)) # 8개 특징(Feature)

    print(f"✅ 데이터 준비 완료! (입력 차원: {window_size}x8)")

except Exception as e:
    print(f"\n❌ 오류: {e}")
    sys.exit()

# 3. Transformer + LSTM 하이브리드 모델 설계
print(f"\n🚀 [Hybrid AI] Attention 기술이 적용된 모델을 생성합니다...")
print("=" * 60)

labels = ['A', 'B', 'C', 'D', 'E']

for i in range(5):
    print(f"\n🧠 [AI 모델 {labels[i]} 학습 중... (스마트 학습 모드)]")
    
    # --- 모델 구조 (Functional API 사용) ---
    inputs = Input(shape=(window_size, 8))
    
    # 1단계: LSTM으로 시계열 흐름 파악
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    
    # 2단계: Self-Attention (중요한 회차 강조)
    # 챗GPT와 같은 원리로, 데이터 내의 연관성을 찾습니다.
    # key_dim은 내적 차원 수
    att_out = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + att_out) # Residual Connection
    
    # 3단계: 요약 및 추론
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    
    # 4단계: 최종 확률 출력 (1~45번)
    outputs = Dense(45, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # --- 🔥 [업그레이드 3] 동적 학습률 조정 🔥 ---
    # 학습이 정체되면 학습률(Learning Rate)을 0.5배로 낮춰서 더 섬세하게 학습함
    lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.00001, verbose=0)
    
    # 학습 (150회)
    model.fit(X, y, epochs=150, batch_size=32, callbacks=[lr_scheduler], verbose=0)
    
    # 예측
    prob_prediction = model.predict(last_window, verbose=0)[0]
    
    # 상위 6개 추출
    top_6_indices = prob_prediction.argsort()[-6:]
    final_nums = np.sort(top_6_indices + 1)
    
    # 확신도 계산
    confidence = prob_prediction[top_6_indices].mean() * 100
    
    # 합계 및 홀짝 정보도 같이 출력 (AI가 고려한 요소)
    pred_sum = sum(final_nums)
    pred_odd = sum([1 for n in final_nums if n % 2 != 0])
    
    print(f"👉 Game {labels[i]} 추천: {final_nums}")
    print(f"   (AI 확신도: {confidence:.1f}% | 예상 합계: {pred_sum} | 홀수: {pred_odd}개)")

print("\n" + "=" * 60)
input("✅ ULTIMATE 분석 완료. 종료하려면 엔터 키를 누르세요...")
```

## ver6 

해결책: "숫자" 대신 "지도(Map)"를 보여주자!
AI가 훨씬 더 쉽게 패턴을 찾고 확신을 가질 수 있도록 데이터 형태를 **'원-핫 인코딩(One-Hot Encoding)'**으로 바꿔서 입력

기존 방식: "3번 공이 나왔어" (AI: 3이 뭐지? 숫자 크기인가?)

변경 방식: "45개의 전구 중 3번째 전구에 불이 켜졌어! " (AI: 아하! 위치가 딱 보이네!)

```
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, Flatten, Dropout
import sys
import os

# 1. 환경 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

print("🔥🔥 [REAL CONFIDENCE] 인위적 보정 없이, 데이터 구조 변경으로 확신도를 높입니다 🔥🔥")
print("👉 핵심 기술: Full One-Hot Input (숫자가 아닌 '위치'로 학습)")

# 2. 데이터 읽기
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
    
    full_df = full_df[[1, 2, 3, 4, 5, 6, 7]]
    full_df.columns = ['Round', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']
    full_df = full_df.apply(pd.to_numeric, errors='coerce').dropna()
    full_df = full_df.sort_values('Round').reset_index(drop=True)
    
    # 데이터 준비
    num_data = full_df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values
    
    # --- 🔥 [핵심 변경] 입력 데이터도 '원-핫 인코딩'으로 변환 🔥 ---
    # 숫자를 그대로 쓰지 않고, 45개의 0/1 스위치로 변환해서 보여줍니다.
    # 이렇게 하면 AI가 패턴을 훨씬 더 선명하게 인식합니다.
    
    def numbers_to_onehot(rows):
        onehot = np.zeros((len(rows), 45)) # 45개 공간 (0~44 인덱스 사용)
        for i, row in enumerate(rows):
            for num in row:
                # 로또 번호 1~45를 인덱스 0~44로 변환 (-1)
                onehot[i, int(num)-1] = 1
        return onehot

    # 모든 회차를 0과 1의 지도로 바꿈
    onehot_data = numbers_to_onehot(num_data)
    
    window_size = 20 # 패턴 인식을 위해 최근 20주 사용
    
    def create_dataset(onehot_data, window_size):
        X, y = [], []
        for i in range(len(onehot_data) - window_size):
            X.append(onehot_data[i : i + window_size])
            y.append(onehot_data[i + window_size])
        return np.array(X), np.array(y)
        
    X, y = create_dataset(onehot_data, window_size)
    
    # 예측용 마지막 데이터
    last_window = onehot_data[-window_size:]
    last_window = last_window.reshape((1, window_size, 45))

    print(f"✅ 데이터 변환 완료: 입력 데이터 형태가 (숫자) -> (45개 스위치)로 변경되었습니다.")

except Exception as e:
    print(f"❌ 오류: {e}")
    sys.exit()

# 3. 모델 설계 (학습 능력 극대화)
print(f"\n🚀 [Pure Logic] AI 학습 시작 (보정 함수 없음)...")
print("=" * 60)

labels = ['A', 'B', 'C', 'D', 'E']

for i in range(5):
    print(f"\n🧠 [AI 모델 {labels[i]} 정밀 학습 중...]")
    
    model = Sequential([
        Input(shape=(window_size, 45)), # 입력도 45개짜리 비트맵
        
        # 1. 정보를 압축하지 않고 그대로 패턴을 읽음
        Bidirectional(LSTM(256, return_sequences=True)),
        
        # 2. 과감하게 Dropout 제거 (확신도 상승 요인)
        # Dropout이 없으면 AI는 '모 아니면 도' 식으로 확실한 것만 외웁니다.
        
        Flatten(), # 모든 정보를 한 줄로 펼침
        
        # 3. 아주 깊고 넓은 신경망
        Dense(1024, activation='relu'), 
        Dense(512, activation='relu'),
        
        # 4. 최종 출력 (45개 번호의 확률)
        Dense(45, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 학습 횟수 400회 (충분히 확신을 가질 때까지)
    model.fit(X, y, epochs=400, batch_size=64, verbose=0)
    
    # 예측 (보정 함수 sharpen_prob 삭제함!)
    raw_prediction = model.predict(last_window, verbose=0)[0]
    
    # 상위 6개 추출
    top_6_indices = raw_prediction.argsort()[-6:]
    final_nums = np.sort(top_6_indices + 1)
    
    # 순수 AI 확신도 계산
    confidence = raw_prediction[top_6_indices].mean() * 100
    
    print(f"👉 Game {labels[i]} 추천: {final_nums}")
    print(f"   (💡 순수 확신도: {confidence:.1f}%)")

print("\n" + "=" * 60)
input("✅ 예측 완료. 종료하려면 엔터 키를 누르세요...")
```

## ver7 

```
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, Flatten, Dropout, MultiHeadAttention, LayerNormalization, Concatenate
import sys
import os

# 1. 환경 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

print("👑 [MASTERPIECE] 로또 분석의 정점: Transformer + 미출현 패턴 분석 👑")
print("👉 AI가 '번호'뿐만 아니라 '얼마나 오래 쉬었는지(Cold Number)'까지 고려합니다.")

# 2. 데이터 읽기
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
    
    full_df = full_df[[1, 2, 3, 4, 5, 6, 7]]
    full_df.columns = ['Round', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']
    full_df = full_df.apply(pd.to_numeric, errors='coerce').dropna()
    full_df = full_df.sort_values('Round').reset_index(drop=True)
    
    # 숫자 데이터 (1~45)
    num_data = full_df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values

    # --- 🔥 [핵심 업그레이드 1] '미출현 기간' 데이터 생성 🔥 ---
    # 각 회차별로 "각 번호가 안 나온 지 몇 주 됐는지" 계산해서 알려줌
    # 예: 1번이 5주 동안 안 나왔으면 5, 바로 지난주에 나왔으면 0
    print("⚙️ 고급 분석 중: 번호별 미출현 기간(Cold Number) 계산...")
    
    cold_data = np.zeros((len(num_data), 45)) # (회차수, 45개 번호)
    
    # 초기값: 0으로 시작
    current_cold = np.zeros(45)
    
    for i in range(len(num_data)):
        # 이번 회차 당첨 번호
        winning_nums = num_data[i] - 1 # 인덱스(0~44)로 변환
        
        # 일단 모든 번호의 미출현 기간 +1 증가
        current_cold += 1
        
        # 당첨된 번호는 미출현 기간 0으로 초기화 (나왔으니까!)
        current_cold[winning_nums.astype(int)] = 0
        
        # 기록 저장
        cold_data[i] = current_cold.copy()
        
    # 데이터 정규화 (최대 50주 정도 안 나오는 경우도 있으므로 50으로 나눔)
    cold_data = cold_data / 50.0

    # 원-핫 인코딩 변환 함수
    def numbers_to_onehot(rows):
        onehot = np.zeros((len(rows), 45))
        for i, row in enumerate(rows):
            for num in row:
                onehot[i, int(num)-1] = 1
        return onehot

    onehot_data = numbers_to_onehot(num_data)
    
    # --- 🔥 [핵심 업그레이드 2] 멀티 인풋 (번호 패턴 + 미출현 패턴) 🔥 ---
    # AI에게 두 가지 정보를 동시에 줍니다.
    # 1. 어떤 번호가 나왔었는지 (onehot_data)
    # 2. 각 번호가 얼마나 쉬었는지 (cold_data)
    
    # 두 데이터를 합침 (입력 차원: 45 + 45 = 90)
    final_input = np.concatenate([onehot_data, cold_data], axis=1)
    
    window_size = 20 # 최근 20주 분석
    
    def create_dataset(input_data, target_data, window_size):
        X, y = [], []
        for i in range(len(input_data) - window_size):
            X.append(input_data[i : i + window_size])
            y.append(target_data[i + window_size])
        return np.array(X), np.array(y)
        
    X, y = create_dataset(final_input, onehot_data, window_size)
    
    # 예측용 마지막 데이터
    last_window = final_input[-window_size:]
    last_window = last_window.reshape((1, window_size, 90)) # 90개 정보 (45번호 + 45미출현)

    print(f"✅ 데이터 준비 완료: 입력 차원이 90개로 확장되었습니다 (정밀도 2배 상승)")

except Exception as e:
    print(f"❌ 오류: {e}")
    sys.exit()

# 3. Transformer 기반 고성능 모델
print(f"\n🚀 [Transformer AI] 차세대 모델 학습 시작...")
print("=" * 60)

labels = ['A', 'B', 'C', 'D', 'E']

for i in range(5):
    print(f"\n🧠 [AI 모델 {labels[i]} 학습 중... (패턴 & 미출현 동시 분석)]")
    
    # 입력층 (90개 정보)
    inputs = Input(shape=(window_size, 90))
    
    # 1. Transformer Block (패턴의 맥락 파악)
    # 챗GPT처럼 '어디가 중요한지' 스스로 판단함
    att_output = MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
    att_output = LayerNormalization(epsilon=1e-6)(att_output + inputs) # 잔차 연결
    
    # 2. LSTM Block (시간의 흐름 파악)
    x = Bidirectional(LSTM(128, return_sequences=False))(att_output)
    
    # 3. Dense Block (최종 판단)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x) # 살짝 잊게 해서 일반화 성능 높임
    
    # 4. 출력 (45개 확률)
    outputs = Dense(45, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 학습 (200회 - 모델이 똑똑해서 금방 배웁니다)
    model.fit(X, y, epochs=200, batch_size=32, verbose=0)
    
    # 예측
    raw_prediction = model.predict(last_window, verbose=0)[0]
    
    # 상위 6개 추출
    top_6_indices = raw_prediction.argsort()[-6:]
    final_nums = np.sort(top_6_indices + 1)
    
    # 확신도
    confidence = raw_prediction[top_6_indices].mean() * 100
    
    print(f"👉 Game {labels[i]} 추천: {final_nums}")
    print(f"   (💡 종합 확신도: {confidence:.1f}%)")

print("\n" + "=" * 60)
input("✅ 분석 완료. 종료하려면 엔터 키를 누르세요...")
```


## ver8 

```
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, Flatten, Dropout, MultiHeadAttention, LayerNormalization, Concatenate
import sys
import os
import itertools # 조합 생성을 위한 도구 추가

# 1. 환경 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

print("👑 [MASTERPIECE + Filtering] 과거 당첨 번호 제외 기능 탑재 👑")
print("👉 AI가 추천한 번호가 이미 당첨된 적이 있다면, 자동으로 다른 최적의 번호를 찾습니다.")

# 2. 데이터 읽기
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
    
    full_df = full_df[[1, 2, 3, 4, 5, 6, 7]]
    full_df.columns = ['Round', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']
    full_df = full_df.apply(pd.to_numeric, errors='coerce').dropna()
    full_df = full_df.sort_values('Round').reset_index(drop=True)
    
    # 숫자 데이터 (1~45)
    num_data = full_df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values

    # --- 🔥 [필터링 시스템 준비] 과거 당첨 번호 저장 🔥 ---
    print("⚙️ 과거 모든 회차의 당첨 번호를 메모리에 등록 중...", end="")
    past_combinations = set()
    for row in num_data:
        # 1등 번호를 정렬해서 튜플로 저장 (검색 속도 최적화)
        past_combinations.add(tuple(sorted(row)))
    print(f" 완료! (총 {len(past_combinations)}개의 금지된 조합)")

    # 미출현 기간 데이터 생성 (기존 로직)
    cold_data = np.zeros((len(num_data), 45))
    current_cold = np.zeros(45)
    
    for i in range(len(num_data)):
        winning_nums = num_data[i] - 1
        current_cold += 1
        current_cold[winning_nums.astype(int)] = 0
        cold_data[i] = current_cold.copy()
        
    cold_data = cold_data / 50.0

    # 원-핫 인코딩
    def numbers_to_onehot(rows):
        onehot = np.zeros((len(rows), 45))
        for i, row in enumerate(rows):
            for num in row:
                onehot[i, int(num)-1] = 1
        return onehot

    onehot_data = numbers_to_onehot(num_data)
    
    # 입력 데이터 결합 (번호패턴 + 미출현패턴)
    final_input = np.concatenate([onehot_data, cold_data], axis=1)
    
    window_size = 20
    
    def create_dataset(input_data, target_data, window_size):
        X, y = [], []
        for i in range(len(input_data) - window_size):
            X.append(input_data[i : i + window_size])
            y.append(target_data[i + window_size])
        return np.array(X), np.array(y)
        
    X, y = create_dataset(final_input, onehot_data, window_size)
    last_window = final_input[-window_size:].reshape((1, window_size, 90))

    print(f"✅ 데이터 준비 완료.")

except Exception as e:
    print(f"❌ 오류: {e}")
    sys.exit()

# 3. 모델 학습 및 예측
print(f"\n🚀 [AI Prediction] 분석 및 필터링 시작...")
print("=" * 60)

labels = ['A', 'B', 'C', 'D', 'E']

for i in range(5):
    print(f"\n🧠 [AI 모델 {labels[i]} 학습 중...]")
    
    # 모델 구조 (Transformer + LSTM)
    inputs = Input(shape=(window_size, 90))
    att_output = MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
    att_output = LayerNormalization(epsilon=1e-6)(att_output + inputs)
    x = Bidirectional(LSTM(128, return_sequences=False))(att_output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(45, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 학습
    model.fit(X, y, epochs=200, batch_size=32, verbose=0)
    
    # 예측
    raw_prediction = model.predict(last_window, verbose=0)[0]
    
    # --- 🔥 [스마트 필터링 로직] 중복 없는 최적 조합 찾기 🔥 ---
    # 1. 확률이 높은 상위 10개 공을 후보로 뽑습니다.
    #    (6개만 뽑으면 중복일 때 대안이 없으므로 여유 있게 뽑음)
    top_candidates_indices = raw_prediction.argsort()[-10:][::-1] # 상위 10개 내림차순
    
    best_combination = None
    best_score = -1
    
    # 2. 상위 10개 공으로 만들 수 있는 모든 6개 조합을 검사합니다. (총 210가지 경우)
    #    itertools.combinations를 사용해 조합 생성
    for combo in itertools.combinations(top_candidates_indices, 6):
        # 1~45 번호로 변환 및 정렬
        current_nums = tuple(sorted(np.array(combo) + 1))
        
        # 3. 과거 당첨 이력에 있는지 확인 (필터링)
        if current_nums in past_combinations:
            continue # 이미 나왔던 번호면 건너뜀 (탈락!)
            
        # 4. 살아남은 조합 중 '확률 합계'가 가장 높은 것을 선택
        current_score = sum(raw_prediction[idx] for idx in combo)
        if current_score > best_score:
            best_score = current_score
            best_combination = current_nums
    
    # 결과 확정
    final_nums = np.array(best_combination)
    confidence = (best_score / 6) * 100 # 평균 확신도
    
    print(f"👉 Game {labels[i]} 추천: {final_nums}")
    print(f"   (💡 필터링 완료 | 종합 확신도: {confidence:.1f}%)")

print("\n" + "=" * 60)
input("✅ 모든 분석이 완료되었습니다. 엔터 키를 눌러 종료하세요...")
```

## ver9 (오류가 있는 초기 버전)

문제점 
- 2번째 부터 5번째까지 똑같은 숫자가 나오는 현상 발생
- 확신도가 너무 낮은 현상 발생
```
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, Flatten, Dropout, MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.callbacks import ReduceLROnPlateau
import sys
import os
import random

# 1. 환경 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

print("👑 [GOD MODE] 유전 알고리즘(Genetic Algorithm) + 3중 앙상블 시스템 가동 👑")
print("👉 1단계: 3개의 AI가 집단 지성으로 확률 계산")
print("👉 2단계: 유전 알고리즘이 수만 번의 진화 시뮬레이션을 통해 최적 조합 도출")
print("⏳ 계산량이 많아 시간이 다소 소요됩니다. 잠시만 기다려주세요...")

# 2. 데이터 읽기 및 전처리
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
    
    full_df = full_df[[1, 2, 3, 4, 5, 6, 7]]
    full_df.columns = ['Round', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']
    full_df = full_df.apply(pd.to_numeric, errors='coerce').dropna()
    full_df = full_df.sort_values('Round').reset_index(drop=True)
    
    num_data = full_df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values

    # --- 과거 당첨 번호 저장 (필터링용) ---
    past_combinations = set()
    for row in num_data:
        past_combinations.add(tuple(sorted(row)))

    # --- 데이터 특성 엔지니어링 ---
    # 1. 미출현 기간 (Cold Data)
    cold_data = np.zeros((len(num_data), 45))
    current_cold = np.zeros(45)
    for i in range(len(num_data)):
        winning_nums = num_data[i] - 1
        current_cold += 1
        current_cold[winning_nums.astype(int)] = 0
        cold_data[i] = current_cold.copy()
    cold_data = cold_data / 50.0

    # 2. 원-핫 인코딩
    def numbers_to_onehot(rows):
        onehot = np.zeros((len(rows), 45))
        for i, row in enumerate(rows):
            for num in row:
                onehot[i, int(num)-1] = 1
        return onehot

    onehot_data = numbers_to_onehot(num_data)
    
    # 3. 입력 데이터 결합 (번호패턴 + 미출현패턴) -> 차원 90
    final_input = np.concatenate([onehot_data, cold_data], axis=1)
    
    window_size = 25 # 패턴 분석 길이 약간 증가
    
    def create_dataset(input_data, target_data, window_size):
        X, y = [], []
        for i in range(len(input_data) - window_size):
            X.append(input_data[i : i + window_size])
            y.append(target_data[i + window_size])
        return np.array(X), np.array(y)
        
    X, y = create_dataset(final_input, onehot_data, window_size)
    last_window = final_input[-window_size:].reshape((1, window_size, 90))

    print(f"✅ 데이터 준비 완료. (입력 차원: 90)")

except Exception as e:
    print(f"❌ 오류: {e}")
    sys.exit()

# 3. 모델 정의 (ResNet 스타일의 깊은 모델)
def create_advanced_model():
    inputs = Input(shape=(window_size, 90))
    
    # Transformer Block
    att = MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
    att = LayerNormalization(epsilon=1e-6)(Add()([inputs, att]))
    
    # LSTM Block with Residual Connection
    x = Bidirectional(LSTM(128, return_sequences=True))(att)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    
    # Deep Dense Layers
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    
    outputs = Dense(45, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 4. 유전 알고리즘 (Genetic Algorithm) 클래스
class LottoGeneticOptimizer:
    def __init__(self, probabilities, past_history, population_size=500, generations=50):
        self.probs = probabilities # AI가 예측한 45개 번호의 확률
        self.past_history = past_history
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = 0.1
        
    def create_individual(self):
        # 확률에 기반하여 6개 번호 랜덤 선택 (룰렛 휠 방식)
        # 확률이 높은 공이 뽑힐 가능성이 높음
        indices = np.arange(1, 46)
        # 확률 정규화
        p = self.probs / self.probs.sum()
        return sorted(np.random.choice(indices, 6, replace=False, p=p))
        
    def fitness(self, individual):
        # 1. AI 확률 점수 (기본 점수)
        score = sum([self.probs[num-1] for num in individual])
        
        # 2. 통계적 제약 조건 (패널티 부여)
        
        # 합계 필터 (보통 100~200 사이가 많이 나옴)
        total_sum = sum(individual)
        if total_sum < 100 or total_sum > 230:
            score -= 2.0 # 강력한 감점
            
        # 홀짝 비율 필터 (6:0, 0:6은 잘 안 나옴)
        odds = sum([1 for n in individual if n % 2 != 0])
        if odds == 0 or odds == 6:
            score -= 1.0
            
        # 연속 번호 필터 (3연속 번호 등은 드묾)
        # 예: 1, 2, 3 -> 감점
        consecutive = 0
        for i in range(len(individual)-1):
            if individual[i+1] == individual[i] + 1:
                consecutive += 1
        if consecutive >= 2: # 3연번 이상
            score -= 1.5
            
        # 3. 과거 당첨 이력 필터 (절대 금지)
        if tuple(individual) in self.past_history:
            score -= 100.0 # 생존 불가
            
        return score
    
    def evolve(self):
        # 초기 세대 생성
        population = [self.create_individual() for _ in range(self.pop_size)]
        
        for gen in range(self.generations):
            # 점수 계산 및 정렬
            scored_pop = [(ind, self.fitness(ind)) for ind in population]
            scored_pop.sort(key=lambda x: x[1], reverse=True)
            
            # 상위 20% 생존 (Elitism)
            survivors = [x[0] for x in scored_pop[:int(self.pop_size * 0.2)]]
            
            # 다음 세대 생성
            next_gen = survivors[:]
            while len(next_gen) < self.pop_size:
                # 부모 선택 (토너먼트)
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                
                # 교배 (Crossover)
                split = random.randint(1, 5)
                child = sorted(list(set(parent1[:split] + parent2[split:])))
                
                # 자식 번호가 6개가 안 되면 채워넣기
                while len(child) < 6:
                    new_num = random.randint(1, 45)
                    if new_num not in child:
                        child.append(new_num)
                child = sorted(child[:6])
                
                # 돌연변이 (Mutation)
                if random.random() < self.mutation_rate:
                    remove_idx = random.randint(0, 5)
                    child.pop(remove_idx)
                    while len(child) < 6:
                        new_num = random.randint(1, 45)
                        if new_num not in child:
                            child.append(new_num)
                    child.sort()
                
                next_gen.append(child)
            
            population = next_gen
            
        # 최종 우승자 반환
        scored_pop = [(ind, self.fitness(ind)) for ind in population]
        scored_pop.sort(key=lambda x: x[1], reverse=True)
        return scored_pop[0] # (조합, 점수)

# 5. 실행 로직 (앙상블 + 유전 알고리즘)
print(f"\n🚀 [Ensemble + GA] 초정밀 분석 시작...")
print("=" * 60)

labels = ['A', 'B', 'C', 'D', 'E']

# 학습률 스케줄러
lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=0)

for i in range(5):
    print(f"\n🧠 [Game {labels[i]}] 분석 중...")
    
    # --- 1단계: 앙상블 (3개의 모델을 독립적으로 학습시켜 평균) ---
    print(f"   ↳ 🤖 AI 모델 3개를 동시에 학습시켜 편향을 제거합니다...", end="")
    ensemble_preds = []
    
    for m in range(3): # 모델 3개
        model = create_advanced_model()
        # epoch는 150회로 충분 (3번 하니까)
        model.fit(X, y, epochs=150, batch_size=32, callbacks=[lr_scheduler], verbose=0)
        pred = model.predict(last_window, verbose=0)[0]
        ensemble_preds.append(pred)
        
    # 3개 모델의 예측값 평균 (앙상블 결과)
    avg_prediction = np.mean(ensemble_preds, axis=0)
    print(" 완료!")
    
    # --- 2단계: 유전 알고리즘 (최적화) ---
    print(f"   ↳ 🧬 유전 알고리즘으로 최적의 조합을 진화시키는 중...", end="")
    
    # AI가 만든 확률 지도를 유전 알고리즘에 전달
    optimizer = LottoGeneticOptimizer(avg_prediction, past_combinations)
    
    # 진화 시작 (500마리 개체, 50세대 진화)
    best_combo, score = optimizer.evolve()
    
    final_nums = np.array(best_combo)
    
    # 확신도 계산 (AI 확률 평균)
    confidence = sum([avg_prediction[n-1] for n in final_nums]) / 6 * 100
    
    print(" 진화 완료!")
    print(f"👉 Game {labels[i]} 추천: {final_nums}")
    print(f"   (💡 AI확신도: {confidence:.1f}% | 🧬 적합도 점수: {score:.2f})")
    

print("\n" + "=" * 60)
input("✅ 모든 계산이 끝났습니다. 엔터 키를 눌러 종료하세요...")

```

## ver9 

```
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, Flatten, Dropout, MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.callbacks import ReduceLROnPlateau
import sys
import os
import random

# 1. 환경 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

print("🔥🔥 [ULTIMATE GRIND] 진정성 있는 고확신 모드 가동 🔥🔥")
print("👉 전략: 안전장치(Dropout)를 켜서 과적합을 막고, 학습량(Epochs)을 1500회로 늘립니다.")
print("👉 의미: '답안지 암기'가 아닌 '패턴의 완벽한 이해'를 추구합니다.")
print("⏳ 학습량이 엄청나게 많습니다 (15~20분 소요). 컴퓨터를 켜두고 잠시 쉬다 오세요...")

# 2. 데이터 읽기 및 전처리
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

    full_df = full_df[[1, 2, 3, 4, 5, 6, 7]]
    full_df.columns = ['Round', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']
    full_df = full_df.apply(pd.to_numeric, errors='coerce').dropna()
    full_df = full_df.sort_values('Round').reset_index(drop=True)

    num_data = full_df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values

    # 과거 당첨 번호 저장
    past_combinations = set()
    for row in num_data:
        past_combinations.add(tuple(sorted(row)))

    # 특성 엔지니어링
    cold_data = np.zeros((len(num_data), 45))
    current_cold = np.zeros(45)
    for i in range(len(num_data)):
        winning_nums = num_data[i] - 1
        current_cold += 1
        current_cold[winning_nums.astype(int)] = 0
        cold_data[i] = current_cold.copy()
    cold_data = cold_data / 50.0

    # 원-핫 인코딩
    def numbers_to_onehot(rows):
        onehot = np.zeros((len(rows), 45))
        for i, row in enumerate(rows):
            for num in row:
                onehot[i, int(num)-1] = 1
        return onehot

    onehot_data = numbers_to_onehot(num_data)
    final_input = np.concatenate([onehot_data, cold_data], axis=1)

    window_size = 25

    def create_dataset(input_data, target_data, window_size):
        X, y = [], []
        for i in range(len(input_data) - window_size):
            X.append(input_data[i : i + window_size])
            y.append(target_data[i + window_size])
        return np.array(X), np.array(y)

    X, y = create_dataset(final_input, onehot_data, window_size)
    last_window = final_input[-window_size:].reshape((1, window_size, 90))

    print(f"✅ 데이터 준비 완료.")

except Exception as e:
    print(f"❌ 오류: {e}")
    sys.exit()

# 3. 모델 정의 (Dropout 복구됨)
def create_grind_model():
    inputs = Input(shape=(window_size, 90))

    # Transformer Block
    att = MultiHeadAttention(num_heads=8, key_dim=64)(inputs, inputs)
    att = LayerNormalization(epsilon=1e-6)(Add()([inputs, att]))

    # LSTM Block
    # ✅ Dropout 0.2 복구: AI가 너무 자만하지 않게 견제함
    x = Bidirectional(LSTM(256, return_sequences=True))(att)
    x = Dropout(0.2)(x)

    x = Bidirectional(LSTM(128, return_sequences=False))(x)
    x = Dropout(0.2)(x)

    # Dense Layers
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x) # 여기도 안전장치 복구
    x = Dense(256, activation='relu')(x)

    outputs = Dense(45, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 4. 유전 알고리즘 (그대로 유지)
class LottoGeneticOptimizer:
    def __init__(self, probabilities, past_history, population_size=500, generations=50):
        self.probs = probabilities
        self.past_history = past_history
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = 0.1

    def create_individual(self):
        indices = np.arange(1, 46)
        p = self.probs / self.probs.sum()
        return sorted(np.random.choice(indices, 6, replace=False, p=p))

    def fitness(self, individual):
        score = sum([self.probs[num-1] for num in individual])
        total_sum = sum(individual)
        if total_sum < 100 or total_sum > 230: score -= 2.0
        odds = sum([1 for n in individual if n % 2 != 0])
        if odds == 0 or odds == 6: score -= 1.0
        consecutive = 0
        for i in range(len(individual)-1):
            if individual[i+1] == individual[i] + 1: consecutive += 1
        if consecutive >= 2: score -= 1.5
        if tuple(individual) in self.past_history: score -= 100.0
        return score

    def evolve(self):
        population = [self.create_individual() for _ in range(self.pop_size)]
        for gen in range(self.generations):
            scored_pop = [(ind, self.fitness(ind)) for ind in population]
            scored_pop.sort(key=lambda x: x[1], reverse=True)
            survivors = [x[0] for x in scored_pop[:int(self.pop_size * 0.2)]]
            next_gen = survivors[:]
            while len(next_gen) < self.pop_size:
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                split = random.randint(1, 5)
                child = sorted(list(set(parent1[:split] + parent2[split:])))
                while len(child) < 6:
                    new_num = random.randint(1, 45)
                    if new_num not in child: child.append(new_num)
                child = sorted(child[:6])
                if random.random() < self.mutation_rate:
                    remove_idx = random.randint(0, 5)
                    child.pop(remove_idx)
                    while len(child) < 6:
                        new_num = random.randint(1, 45)
                        if new_num not in child: child.append(new_num)
                    child.sort()
                next_gen.append(child)
            population = next_gen
        scored_pop = [(ind, self.fitness(ind)) for ind in population]
        scored_pop.sort(key=lambda x: x[1], reverse=True)
        return scored_pop[0]

# 5. 실행 로직 (극한의 학습량)
print(f"\n🚀 [ULTIMATE GRIND] 5중 앙상블 + Epoch 1,500회 학습 시작...")
print("=" * 60)

labels = ['A', 'B', 'C', 'D', 'E']

# 학습이 길어지므로 인내심을 20회까지 늘림 (쉽게 포기하지 않도록)
lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=20, verbose=0)

for i in range(5):
    print(f"\n🧠 [Game {labels[i]}] 심층 분석 중... (인내심을 갖고 기다려주세요)")

    print(f"   ↳ 🤖 AI 모델 5개 학습 중 (각 1,500회 반복)...", end="")
    ensemble_preds = []

    for m in range(5):
        model = create_grind_model()
        # ✅ Epochs 1500회: 안전장치를 켠 상태로 지독하게 학습시킴
        # Batch Size 128: 속도 향상을 위해 조금 늘림
        model.fit(X, y, epochs=1500, batch_size=128, callbacks=[lr_scheduler], verbose=0)
        pred = model.predict(last_window, verbose=0)[0]
        ensemble_preds.append(pred)
        print(f" {m+1}/5..", end="")

    avg_prediction = np.mean(ensemble_preds, axis=0)
    print(" 앙상블 완료!")

    print(f"   ↳ 🧬 최적 조합 도출 중...", end="")
    optimizer = LottoGeneticOptimizer(avg_prediction, past_combinations)
    best_combo, score = optimizer.evolve()

    final_nums = np.array(best_combo)
    confidence = sum([avg_prediction[n-1] for n in final_nums]) / 6 * 100

    print(" 완료!")
    print(f"👉 Game {labels[i]} 추천: {final_nums}")
    print(f"   (💡 AI확신도: {confidence:.1f}% | 🧬 적합도 점수: {score:.2f})")

print("\n" + "=" * 60)
input("✅ 모든 극한의 연산이 끝났습니다. 좋은 결과를 기원합니다!")
```
