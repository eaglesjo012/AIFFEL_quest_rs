# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 조영근
- 리뷰어 : 박희지


# PRT(Peer Review Template)
- [x]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    1) 번역기 모델 학습에 필요한 텍스트 데이터 전처리가 한국어 포함하여 잘 이루어졌다.
       - preprocess_sentence 함수(전처리 셀)에서 lower(), 구두점/특수문자 정규식 처리, Mecab morphs() 형태소 분석을 모두 확인하였다.
       - zip(raw_kor, raw_eng)로 쌍을 먼저 묶은 뒤 set()으로 중복을 제거해, 병렬 쌍이 흐트러지는 오류를 피했다.

         <img width="637" height="546" alt="image" src="https://github.com/user-attachments/assets/752dbabc-1ddd-435f-be79-e3ab75198b11" />
         <img width="1109" height="816" alt="image" src="https://github.com/user-attachments/assets/b573e454-f8ab-48d3-bc0d-1fff5907c056" />
         
    2) Attentional Seq2seq 모델이 정상적으로 구동된다.
       - 훈련에서 50에폭 전 구간의 Train/Validation Loss가 확인된다. Train Loss 5.7342 → 1.8955, Validation Loss 7.5216 → 3.7748로 발산 없이 단조적으로 감소하였다.
       - Loss 시각화 셀에 실제 그래프 이미지가 저장되어 있어 추세를 시각적으로도 확인하였다.
      
         <img width="915" height="603" alt="image" src="https://github.com/user-attachments/assets/8e9d0575-5072-4b2b-af95-7cdb0eaf57fb" />

    3) 테스트 결과 의미가 통하는 수준의 번역문이 생성되었다.
       - 예문 4개의 번역 결과에서 핵심 단어를 포착하였다. 문장 구조가 반복적이고 비문법적인 부분이 있긴 하지만 의미가 통하는 수준은 달성되었다.
      
         <img width="563" height="183" alt="image" src="https://github.com/user-attachments/assets/8b2295d4-86c2-42af-8da8-f8fdfe912dd1" />


    
- [x]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - `translate_sentence` 함수가 가장 복잡하다고 생각한다.
      - 단순 greedy decoding에 그치지 않고, 반복 생성 문제를 완화하기 위한 휴리스틱(최근 3단어 반복 패널티, 직전 단어 연속 반복 강력 금지, 문장 길이에 따른 <end> 토큰 강제 유도)이 겹겹이 적용되어 있어 로직의 분기가 많고 각 패널티 값의 의도를 모르면 코드만 보고 이해하기 어렵기 때문이다.
      - 각 휴리스틱 블록마다 "# 1. 최근 3단어 반복 패널티(A-B-A 패턴 방지)"처럼 번호와 목적이 함께 달린 인라인 주석이 있어, 코드만으로는 알기 어려운 "왜 이 로직이 필요한가"까지 파악할 수 있어서 이해하기 쉬웠다.
     
      <img width="885" height="1154" alt="image" src="https://github.com/user-attachments/assets/4ef09f8c-ea9a-4c53-9038-c70fced28c49" />

        
- [x]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    - `[수정된]` 태그로 실험/개선 이력이 남아 있다. 각 수정 지점마다 왜 그렇게 바꿨는지 이유가 주석으로 남아 있어 실험적 개선 과정임을 알 수 있었다.
      - 인코더를 단방향 -> 양방향 GRU로 전환하였고, Dropout을 추가하였다.
      - OOV 문제를 완화하기 위해 어휘 크기를 10,000 -> 20,000으로 증가하였다.
      - Mixed Precision(AMP) 학습을 도입하고, ReduceLROnPlateau 스케줄러를 적용하였다.
      - 디코딩 단계에서 반복 억제 및 길이 제어 로직을 추가하였다.

     <img width="746" height="504" alt="스크린샷 2026-09-10 103539" src="https://github.com/user-attachments/assets/e692cc5d-6988-41f6-abe6-a6b2285393ba" />
     <img width="1056" height="400" alt="image" src="https://github.com/user-attachments/assets/a11238cb-7889-4e95-91ad-571973771a95" />

        
- [ ]  **4. 회고를 잘 작성했나요?**
    - 회고와 전체 코드 실행 플로우는 없었다.
        
- [x]  **5. 코드가 간결하고 효율적인가요?**
    - 인코더/디코더/어텐션/훈련/평가/번역 함수가 역할별로 분리되어 있어 모듈화가 잘 되었다.
    - 정규식을 루프 밖에서 미리 컴파일해 전처리 속도를 최적화하였다.


# 회고(참고 링크 및 코드 개선)
```
이번 코드를 보면서 어텐션 기반 Seq2seq를 PyTorch로 직접 구현할 때 인코더 hidden state를 양방향 GRU와 맞추기 위해 어떻게 shape을 조정하는지 배울 수 있었습니다.
특히 translate_sentence 함수에서 반복 생성을 막기 위해 패널티를 여러 단계로 겹쳐 적용한 부분이 인상 깊었습니다.
`[수정됨]` 주석들을 따라가면서 하이퍼파라미터를 하나씩 바꿔본 흐름을 볼 수 있었던 것도 도움이 됐습니다.
```
