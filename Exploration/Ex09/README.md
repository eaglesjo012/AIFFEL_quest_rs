# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 조영근
- 리뷰어 : 강지수


# PRT(Peer Review Template)

- [x] **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 한국어 Q-A 데이터 수집 및 전처리, SentencePiece 토큰화, Teacher Forcing용 데이터셋 구성, Encoder-Decoder Transformer 구현, 학습 및 자기회귀 추론까지 전체 파이프라인이 구현되어 있습니다.
    - 원본 11,823개의 Q-A 데이터에서 결측값과 중복 질문-답변 쌍을 제거하고 학습/검증 데이터를 구성하였습니다.
      <img width="571" height="373" alt="스크린샷 2026-08-20 오후 5 21 20" src="https://github.com/user-attachments/assets/1e7f52dc-d4cc-4c8d-8506-041642e0e653" />
    - `"안녕하세요" → "안녕하세요."`, `"고마워" → "감사합니다."`, `"요즘 힘들어" → "언젠가 다 잘할 수 있을 거예요."`와 같이 맥락상 자연스러운 응답이 생성된 사례가 있습니다.
    - 반면 `"오늘 기분이 어때?" → "사랑에 나이는 중요하지 않아요."`처럼 일부 입력에서는 질문과 답변의 의미 연결이 약한 사례도 확인되었습니다.
    - 최종 Best Validation Token Accuracy는 약 `0.4682`로 설정한 목표 `0.50`에는 도달하지 못했지만, 학습 결과와 한계가 모두 기록되어 있어 모델의 실제 성능을 확인할 수 있었습니다.
      <img width="828" height="78" alt="스크린샷 2026-08-20 오후 5 20 40" src="https://github.com/user-attachments/assets/e2e34555-ba9d-44c0-9492-17317e185ffb" />


- [x] **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - Transformer의 핵심 요소인 `padding_mask`, `look_ahead_mask`, `scaled_dot_product_attention`, `MultiHeadAttention`, `EncoderLayer`, `DecoderLayer`가 각각 함수와 클래스로 나누어져 있어 전체 구조를 따라가기 좋았습니다.
    - Encoder Self-Attention, Decoder Masked Self-Attention, Cross-Attention이 코드상 명확하게 구분되어 있다는 점도 좋았습니다.
      <img width="1077" height="367" alt="스크린샷 2026-08-20 오후 5 23 34" src="https://github.com/user-attachments/assets/e0ec480a-0689-452e-9799-1878639a7903" />


- [x] **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나 새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 기존 학습에서는 Validation Loss가 약 Epoch 12에서 `3.8848`까지 감소한 뒤 마지막에는 `4.0402`까지 다시 상승하는데도 Train Loss는 계속 감소하여, 이를 과적합으로 판단하고 학습 전략을 단계적으로 수정하였습니다.
    - 첫 번째 개선에서는 단순히 epoch를 더 늘리는 대신 `best checkpoint`, Early Stopping, `ReduceLROnPlateau`, Label Smoothing 등을 도입하고 `MAX_LENGTH 40→48`, `FF_DIM 256→512`로 변경하였습니다. 또한 PAD를 제외한 실제 토큰 기준으로 loss/accuracy를 집계하고, 추론 시 반복 2-gram을 억제하는 방식까지 추가하였습니다.
    - 성능 개선 보고서의 `첫번째`와 `두번째` 설정 블록은 동일한 하이퍼파라미터로 기록되어 있어 두 실행 사이의 별도 설정 변경은 확인하기 어려웠습니다. 대신 최종 세 번째 설정에서는 평가 목표 자체를 보다 명확하게 변경한 점이 눈에 띄었습니다.
    - 최종 실험에서는 `TARGET_VAL_ACC=0.50`을 설정하고 Epoch를 `30→60`, Learning Rate를 `5e-4→3e-4`, Dropout을 `0.15→0.10`, Label Smoothing을 `0.05→0.0`으로 변경했습니다. 또한 Early Stopping을 비활성화하고, 모델 선택 기준을 Validation Loss 최소에서 **Validation Accuracy 우선, Validation Loss 보조** 방식으로 변경하였습니다.
    - 그 결과 Validation Loss 자체는 Epoch 12에서 `3.8121`로 가장 낮았지만 Validation Accuracy는 이후에도 상승하여, Epoch 46에서 가장 높은 `val_acc=0.4682`를 기록한 모델을 최종 checkpoint로 복원하였습니다.
    - 목표로 설정한 `val_acc=0.50`에는 도달하지 못했지만, 하나의 지표만 보고 학습을 종료하지 않고 Loss와 Accuracy의 움직임을 비교하면서 모델 선택 기준까지 수정한 실험 과정이 잘 기록되어 있습니다.

    - 추가로 비교해 보면 좋을 것 같은 부분은 **Best Validation Loss 모델과 Best Validation Accuracy 모델의 실제 생성 품질 비교**입니다.
    - 현재 최종 실험에서는 Epoch 12 모델이 Loss 측면에서는 가장 좋고, Epoch 46 모델이 Token Accuracy 측면에서는 가장 좋습니다.
      <img width="790" height="555" alt="스크린샷 2026-08-20 오후 5 24 23" src="https://github.com/user-attachments/assets/2c6cdd22-8354-4935-8f5a-96bf95c4b6bf" />



- [x] **4. 회고를 잘 작성했나요?**
    - 회고에서 한국어 대화 전처리 → SentencePiece → Encoder-Decoder Transformer 학습이라는 프로젝트의 핵심 흐름을 정리하고 있습니다.
    - 특히 학습 시 Teacher Forcing과 실제 추론 시 자기회귀 생성의 차이를 직접 구현한 경험을 남긴 점이 좋았습니다.
    - 프로젝트 본문에는 과적합 문제와 이를 개선하기 위한 실험 기록이 상세히 남아 있어 단순한 최종 성능보다 모델을 학습하면서 무엇을 관찰하고 변경했는지를 확인할 수 있었습니다.

- [x] **5. 코드가 간결하고 효율적인가요?**
    - 데이터 전처리는 `clean_text`, 데이터셋은 `ChatbotDataset`, Transformer 내부는 Attention / Encoder / Decoder 클래스로 분리되어 있어 구성 요소별 역할을 구분하기 쉽습니다.

# 회고(참고 링크 및 코드 개선)

기존에는 Validation Loss가 약 12 epoch 이후 다시 증가하는데도 마지막 epoch 모델을 사용했지만,
이후 Best Checkpoint, Early Stopping, ReduceLROnPlateau, Label Smoothing 등을 적용해
일반화 성능이 좋은 시점의 모델을 선택하려고 한 과정이 잘 기록되어 있었습니다.

이후 최종 실험에서는 목표를 `val_acc=0.50`으로 명확히 설정하고,
Validation Loss가 아니라 Validation Accuracy를 우선하여 best checkpoint를 선택하도록 평가 전략 자체를 변경한 점이 흥미로웠습니다.
