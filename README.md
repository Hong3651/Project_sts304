# 🏭 STS (Stainless Steel) Defect Prediction Project

> **"ROSE-SMOTE 기반의 불균형 데이터 보정 및 공정 이상 조기 예측 모델"**

![Python](https://img.shields.io/badge/Python-3.13.7-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Validation-brightgreen)

---

## 1. 프로젝트 개요 (Overview)

### 목적
본 프로젝트는 **STS(스테인리스강) 제조 공정 데이터**를 기반으로 불량 발생 여부를 예측하기 위한 머신러닝 분석 프로젝트입니다. 다단계 공정에서 수집된 대규모 데이터를 통합·정제하고, **40:1의 극심한 데이터 불균형**을 해결하여 공정 이상을 조기에 식별하는 것을 목표로 합니다.

### 문제 인식 (Problem Statement)
- **데이터 불균형**: 종속변수(`judge`)의 클래스 비율이 **양품:불량 ≈ 40:1**로 불균형함.
- **예측 편향**: 일반적인 학습 시 모델이 다수 클래스인 양품(0)으로만 예측하려는 경향 발생.
- **기존 한계**: 단순 SMOTE 적용만으로는 종합 성능(Recall, F1-score) 향상에 한계가 있음.

---

## 2. 데이터 분석 및 해결 접근 (Approach)

### 2.1 불균형 데이터 해소: ROSE-SMOTE & Native Categorical
불균형 문제를 완화하고 범주형 변수의 중요도 왜곡을 방지하기 위해 두 가지 전략을 수립하여 비교했습니다.

- **전략 A (One-Hot + SMOTE)**: 전통적인 원-핫 인코딩 후 SMOTE 적용.
- **전략 B (Native Cat + ROS) [채택]**: 
    - LightGBM의 **Native Categorical** 기능을 활용해 범주형 변수를 원본 그대로 학습 (변수 중요도 왜곡 방지).
    - **Random Over Sampling (ROS)**을 통해 소수 클래스(불량)를 복제하여 균형을 맞춤.
    - **Permutation Importance**의 그룹합 방식을 적용해 변수 중요도 신뢰성 확보.

### 2.2 임계값 조정 (Threshold Tuning)
기본 임계값(0.5)에서는 불량 검출력이 떨어지므로, 아래 조건을 만족하는 **최적의 임계값**을 탐색했습니다.

이를 통해 **불량 검출 능력(Recall)**과 **판별 효율(F1)**을 동시에 확보했습니다.

---

## 3. 기대 효과 (Expected Outcome)

1. **모델 신뢰성 확보**: 불균형 환경에서도 일반화된 성능을 내는 모델 구축.
2. **공정 최적화**: Vital Few 인자 도출을 통한 공정 제어 최적화.
3. **Zero-Defect 기여**: M형 결함의 사전 예방 및 품질 비용 절감.

---

## 4. 개발 환경 (Environment)

- **기간**: 2025.11.01 ~ 2025.11.14
- **Language**: Python 3.13.7
- **Cloud**: Google Cloud Platform
- **Key Libraries**:
    - `scikit-learn`
    - `lightgbm` / `xgboost`
    - `imbalanced-learn` (SMOTE/ROSE)
    - `pandas`, `numpy`

## 5. 설치 및 실행 (Installation)

```bash
# 필수 라이브러리 설치
pip install -r requirements.txt
