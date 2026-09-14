# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 조영근
- 리뷰어 : 박희지


# PRT(Peer Review Template)
- [x]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    1) 데이터 증강 후 25,093쌍의 데이터셋을 구축하였다.
       - 원본 11,823쌍 → 정제·중복 제거 후 7,681쌍 → Word2Vec 자체학습 기반 Lexical Substitution + 역번역 + EDA 적용 후 23,043쌍 → SentencePiece 토큰화 이후 최종 25,093쌍까지 증강하였다.
       - 정규식을 통한 특수문자/노이즈 제거 및 Mecab을 이용한 형태소 토큰화를 수행하였다.
       - 문장 길이 제한(max_len=40)과 중복 데이터 제거로 고품질의 데이터를 확보했다.
      
         <img width="1095" height="804" alt="image" src="https://github.com/user-attachments/assets/38b0ac20-58cb-4e88-8b09-162f37bc3d29" />
         <img width="669" height="903" alt="image" src="https://github.com/user-attachments/assets/e2014f26-ec7e-4661-a188-a2a12e997f42" />

    2) 과적합을 피할 하이퍼파라미터 셋을 제시하였다.
       - 데이터 규모에 맞춰 `n_layers=2, d_model=256, d_ff=1024`로 경량화하고, Early Stopping을 적용해 best_val_loss 기준으로 체크포인트를 저장했다.
       - 하이퍼파라미터 튜닝 히스토리를 주석으로 남겨 시행착오를 확인할 수 있었다.

         <img width="914" height="859" alt="image" src="https://github.com/user-attachments/assets/ba67764b-c440-4dc0-9a0e-222e2ad007f0" />

    3) 주어진 예문을 포함하여 사용자의 질문에 그럴듯하게 답하였다.
       - Transformer뿐만 아니라 KoGPT2 파인튜닝 + Top-p/Top-k 샘플링까지 예문 4개에 대한 답변을 모두 제출했다.
       - 자체 Transformer의 실제 응답 품질은 아쉽지만 KoGPT2의 응답 품질은 뛰어나, 두 모델 간 품질 격차가 크다는 점을 알 수 있었다.

         <img width="521" height="210" alt="image" src="https://github.com/user-attachments/assets/ddc2541d-2d32-4fdf-89ad-80f455b9a3a5" />
         <img width="464" height="245" alt="image" src="https://github.com/user-attachments/assets/9ab47dce-8285-4fc0-957c-c76d4ae076f8" />



    
- [x]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
      
  - 가장 핵심적이라고 생각하는 코드는 `SentencePiece` 벡터화이다.
     - 이미 형태소 단위로 쪼갠 토큰을 다시 SentencePiece에 넣으면, 형태소 분리 + 서브워드 분리가 중첩되어 토큰이 과도하게 잘게 쪼개질 수 있다. Mecab 결과를 공백으로 재결합한 뒤 SPM에 넣는 방식을 택해 이 문제를 피한 점을 알 수 있었다.
     - `<start>/<end>`를 텍스트가 아닌 SPM의 `bos_id/eos_id`로 처리하도록 바꾼 이유가 명시되어 있어, 왜 이런 설계를 택했는지 리뷰어 입장에서 바로 파악됐다.
    
       <img width="790" height="876" alt="image" src="https://github.com/user-attachments/assets/e06d97e7-147d-473f-9318-285c4d2441de" />


- [x]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
   - `cgi` 모듈이 제거되어 `googletrans`가 깨지는 문제를 `sys.modules`에 목(mock) 모듈을 주입해 해결했다. 원인과 해결 방식이 주석으로 명확히 남아 있다.
   - `googletrans` 제거 후 `h11/httpx/httpcore/huggingface_hub/transformers` 버전 충돌을 막기 위해 관련 모듈 캐시를 강제로 무효화(`importlib.invalidate_caches()` + `sys.modules`에서 삭제)하는 처리를 남겼다. 라이브러리 간 의존성 충돌을 해결하였다.

     <img width="618" height="249" alt="image" src="https://github.com/user-attachments/assets/bb862aa6-754c-4fdf-8528-fbd4d24155f0" />

   - Transformer에 그치지 않고 KoGPT2 파인튜닝을 별도로 시도했다.
     - Greedy, Top-p/Top-k 샘플링, Beam Search 세 가지 방식으로 각각 응답을 생성하고 BLEU를 측정해, 자체 Transformer 결과와 나란히 비교할 수 있게 구성했다.
     - KoGPT2 Greedy 방식에서는 BLEU 1.000(완전 일치)이 두 문장에서 나왔고, Beam Search 평균 BLEU도 0.45 수준으로, 자체 Transformer(Beam 평균 0.05 내외)보다 확연히 높았다.
     
     <img width="650" height="1072" alt="image" src="https://github.com/user-attachments/assets/d7efeeaa-705b-482c-b876-c740b8d9f355" />


        
- [ ]  **4. 회고를 잘 작성했나요?**
    - 회고와 전체 코드 실행 플로우는 없었다.
        
- [x]  **5. 코드가 간결하고 효율적인가요?**
    - `preprocess_sentence`, `build_corpus`, `lexical_sub_list`, `make_corpus_spm` 등 전처리/증강 단계 함수화가 잘 되어 재사용 가능하다.
    - 전역 변수명(`N_LAYERS, D_MODEL, EPOCHS` 등)이 대문자 상수 관례를 따르고 있어 하이퍼파라미터 위치를 파악하기 쉽다.


# 회고(참고 링크 및 코드 개선)
```

```
