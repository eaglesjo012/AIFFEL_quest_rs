# AIFFEL Campus Online Code Peer Review Template
- 코더 : 조영근
- 리뷰어 : 강지수


# PRT(Peer Review Template)

- [x] **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - KoGPT2 기반의 SFT, RM, PPO 학습 흐름과 각 단계의 inference 결과를 확인할 수 있었습니다.
    - 기본 프로젝트 수행에 그치지 않고 Greedy / Beam Search / Top-K / Top-P 등의 decoding 전략 비교와 데이터 정제, KoAlpaca 데이터 추가 학습까지 수행했습니다.
    - 프로젝트 최종 요약을 앞부분에 배치해 전체 실험의 목적과 결과를 한눈에 파악할 수 있었습니다.
      <img width="727" height="185" alt="스크린샷 2026-09-18 오전 10 28 07" src="https://github.com/user-attachments/assets/9228046b-cebd-4e89-955d-1a93316ee424" />


- [x] **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - decoding 실험에서는 `generate_with_strategy()`와 같은 공통 함수를 사용해 전략별 차이를 비교하기 쉽게 구성한 점이 인상적이었습니다.
    <img width="709" height="614" alt="스크린샷 2026-09-18 오전 10 28 56" src="https://github.com/user-attachments/assets/d462efa2-6877-43da-a955-954dd3b33466" />
    - Ablation Study의 목적을 코드 실행 전에 Markdown으로 설명해 어떤 변수를 통제하고 무엇을 관찰하려는 실험인지 이해하기 쉬웠습니다.
    <img width="744" height="140" alt="스크린샷 2026-09-18 오전 10 29 45" src="https://github.com/user-attachments/assets/c9bec1e8-d7ba-4595-8bfe-adc148d74055" />

- [x] **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나 새로운 시도 또는 추가 실험을 수행해봤나요?**
    - Temperature, Repetition Penalty, N-gram Penalty, Beam 수, Length Penalty 등을 변경하는 Ablation Study를 수행해 생성 하이퍼파라미터의 영향을 비교했습니다.
    <img width="719" height="276" alt="스크린샷 2026-09-18 오전 10 30 42" src="https://github.com/user-attachments/assets/a002dc16-8b70-40e6-a402-50145a5f0881" />

    - 기존 SFT 데이터에서 짧은 completion을 제거하고 KoAlpaca 데이터 3,000개를 추가하여 새로운 통합 학습 데이터를 만드는 추가 실험도 수행했습니다.
    <img width="659" height="261" alt="스크린샷 2026-09-18 오전 10 31 21" src="https://github.com/user-attachments/assets/06e7ab8a-7453-45c7-bbff-23f9a5dc13e1" />

- [ ] **4. 회고를 잘 작성했나요?**
    - 프로젝트 앞부분의 `프로젝트 최종 요약`에서 수행한 실험과 주요 결과는 잘 정리되어 있었습니다.

- [x] **5. 코드가 간결하고 효율적인가요?**
    - 반복적으로 사용되는 generation과 tokenization 로직을 함수화하고, SFT dataset 및 data collator를 클래스 형태로 구성하여 반복 코드를 줄였습니다.
    - 여러 decoding 전략을 공통 함수에 parameter만 전달하는 방식으로 비교하여 실험 조건을 변경하기 쉽게 만든 점이 좋았습니다.


# 회고(참고 링크 및 코드 개선)
저는 시간이 부족해서 데이터를 추가할 생각을 못했지만, 영근님은 실험 과정 중 KoAlpaca 데이터를 추가하고, decoding 단계에서는 Temperature, Repetition Penalty, Beam Search 등 여러 조건을 직접 비교해서 다양한 실험의 전개를 수행하셨습니다. 
