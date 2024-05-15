import file_name as fn
import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# 결측치를 대체하는 전처리
from sklearn.impute import SimpleImputer

# 데이터 불러오기
merged_df = pd.read_csv(fn.merge_path)

# 이진 분류를 위한 sleep_efficiency_binary 열 생성
merged_df['sleep_efficiency_binary'] = merged_df['sleep_score'].apply(lambda x: 1 if x >= 85 else 0)
merged_df.drop(['user', 'num'], axis=1, inplace=True)

# 특성과 타겟 분리
X = merged_df.drop(['sleep_score', 'sleep_efficiency_binary'], axis=1)
y = merged_df['sleep_efficiency_binary']

# 훈련 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SimpleImputer를 사용하여 결측치를 중앙값(median)으로 대체합니다.
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# 모델 정의
models = {
    "TPOT": TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB()
}

# 결과를 저장할 딕셔너리 초기화
results = {"Model": [], "Accuracy": [], "F1 Score": [], "Kappa": [], "MCC": []}

# 모델별로 훈련하고 평가
for name, model in models.items():
    # 모델 훈련
    # model.fit(X_train, y_train)
    # 예측
    # y_pred = model.predict(X_test)

    # 모델 훈련
    model.fit(X_train_imputed, y_train)
    # 예측
    y_pred = model.predict(X_test_imputed)

    # 평가 지표 계산
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # 결과 저장
    results["Model"].append(name)
    results["Accuracy"].append(accuracy)
    results["F1 Score"].append(f1)
    results["Kappa"].append(kappa)
    results["MCC"].append(mcc)

# 결과 출력
results_df = pd.DataFrame(results)
print(results_df)
