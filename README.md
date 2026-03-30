# 🎬 Movie Review Sentiment Classification

## 1. Overview
본 프로젝트는 **네이버 부스트캠프 AI Tech 과정**에서 진행한 프로젝트로,  
영화 리뷰 데이터를 기반으로 감성을  
**강한 긍정 / 약한 긍정 / 약한 부정 / 강한 부정**으로 분류하는 것을 목표로 합니다.

단순 긍/부정이 아닌, **미세한 감성 차이를 구분하는 것이 핵심 과제**였습니다.


## 2. Problem

- 4개 클래스 간 **데이터 불균형 존재**
- 약한/강한 감성 간 **경계가 모호함**
- 제한된 사전학습 모델만 사용 가능 (모델 선택 제약 존재)

## 3. Constraints

본 프로젝트는 아래와 같은 규정 하에서 진행되었습니다.

- Hugging Face Hub의 원본 pretrained 모델만 사용 가능
- 허용 모델:
  - klue/roberta-base
  - klue/bert-base
  - kykim/bert-kor-base
  - beomi/kcbert-base
  - monologg/koelectra-base-v3-discriminator
- 파인튜닝된 외부 가중치 및 변형 모델 사용 금지

👉 따라서 **제한된 모델 내에서 성능을 극대화하는 전략이 필요했습니다.**

## 4. Approach (Hypothesis)

### 가설 1  
→ 클래스 불균형이 성능 저하의 주요 원인일 것이다  
→ Loss function 변경이 효과적일 것이다  

### 가설 2  
→ 모델마다 서로 다른 error pattern을 가질 것이다  
→ Ensemble을 통해 성능을 개선할 수 있을 것이다  

## 5. Experiments

### 5.1 Data Analysis & Preprocessing

- 단어 빈도 분석을 통해 상위 단어 확인
- 감성에 영향을 주지 않는 단어를 추가로 제거 (불용어 확장)

👉 결과: baseline 대비 소폭 성능 향상

### 5.2 Loss Function (Focal Loss)

- 클래스 불균형 문제 해결을 위해 Focal Loss 적용

👉 결과: 기대 대비 성능 개선 효과 미미  

👉 해석:
- 클래스 imbalance보다  
  **문맥 이해 부족이 더 큰 문제라고 판단**

### 5.3 Model Selection

- 허용된 모델 내에서 성능 비교
- 가장 성능이 높은 모델을 baseline으로 설정

### 5.4 Ensemble Strategy

- 여러 모델 조합을 통해 ensemble 수행
- weighted ensemble 적용

```python
final_output = w1 * model1 + w2 * model2 + ... 
```

다양한 weight 조합 실험
미세한 weight 변화(0.01 단위)에서도 성능 차이 발생 확인

👉 결과: 단일 모델 대비 성능 크게 향상

### 5.5 K-Fold Cross Validation (실패 경험)
모델 일반화 성능 향상을 위해 K-Fold 적용 시도
하지만 학습 시간 증가 및 서버 자원 부족으로 인해 전체 실험 진행 실패

👉 Insight: 단순 성능 향상 기법보다 환경 제약을 고려한 실험 설계가 중요함을 체감

### 5.6 TAPT (Task Adaptive Pretraining)
도메인 적응을 위해 TAPT 적용

👉 결과: 추가적인 성능 향상 확인

👉 한계:
외부 코드를 활용하여 적용했으며,
내부 구조에 대한 이해는 부족한 상태

## 6. Result
Private Leaderboard: 12 / 219 (상위 5%)
Public Leaderboard: 24 / 219

## 7. Key Insights
✔ Insight 1

Loss function 변경보다
👉 Ensemble 전략이 성능에 더 큰 영향을 미침

✔ Insight 2

모델 성능보다
👉 모델 간 보완성이 중요

✔ Insight 3

제한된 모델 환경에서도
👉 실험 전략을 통해 성능 개선이 가능함을 확인

✔ Insight 4

K-Fold 실패 경험을 통해
👉 자원 제약을 고려한 실험 설계의 중요성 인지

---

## 8. What I Learned
성능 개선은 단순 기법 적용보다 문제 구조를 이해하는 것이 중요함을 학습
다양한 실험을 통해 가설 → 검증 → 해석 과정의 중요성 체감
환경 제약 속에서도 최적의 전략을 찾는 경험 수행

## 9. Collaboration
실험 결과를 팀원들과 공유하여 성능 개선에 기여
경쟁보다 협업을 통해 더 나은 결과 도출

## 10. Tech Stack
Python
PyTorch
Hugging Face Transformers
BERT / RoBERTa

---

## 11. Structure
```
├── sentiment.ipynb
├── TAPT.ipynb
├── run_mlm.py
├── README.md
