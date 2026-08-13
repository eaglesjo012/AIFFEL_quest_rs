# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 조영근
- 리뷰어 : 김시온


# PRT(Peer Review Template)
- [X]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
        - 중요! 해당 조건을 만족하는 부분을 캡쳐해 근거로 첨부
     1. 코퍼스 분석, 전처리, SentencePiece 적용, 토크나이저 구현 및 동작이 빠짐없이 진행되었는가?
     2. SentencePiece 토크나이저가 적용된 Text Classifier 모델이 정상적으로 수렴하여 80% 이상의 test accuracy가 확인되었는가?
     3. SentencePiece 토크나이저를 활용했을 때의 성능을 다른 토크나이저 혹은 SentencePiece의 다른 옵션의 경우와 비교하여 분석을 체계적으로 진행하였는가?

     1. NSMC 데이터셋을 불러온 후 전처리하는 과정이 구현되어 있고 SentencePiece 적용, 토크나이저 구현까지 빠짐없이 포함되어있어 동작이 진행되는것을 확인할 수 있었다.
     2. <img width="248" height="25" alt="image" src="https://github.com/user-attachments/assets/4c5c3ac6-eb6e-41ad-b499-1e53c5f7813b" />
     3. <img width="776" height="126" alt="image" src="https://github.com/user-attachments/assets/8dad5579-045c-412b-bdbb-b8c31a189b48" />

     -> 모든 조건을 만족하면서 주어진 문제를 해결하는 코드가 완성되었다.

    
- [X]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭을 왜 핵심적이라고 생각하는지 확인
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드의 기능, 존재 이유, 작동 원리 등을 기술했는지 확인
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부

     <img width="506" height="598" alt="image" src="https://github.com/user-attachments/assets/7419ae85-85c1-4ebb-b632-4b60d7c2634b" />
     -> ONFIG 부분에서 vocabulary size, model type, epoch, learning rate, max length 등의 실험 조건을 한 곳에서 관리하고 있어 각 실험에서 어떤 설정을 사용하는지 이해하기 쉽게 코드가 작성되어있어 실험결과의 정확도/신뢰성을 높여주는 핵심적인 코드라고 생각이 든다.

      
        
- [X]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 프로젝트 평가 기준에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
     
    기본SentencePiece unigram 8,000 vocabulary뿐만 아니라,unigram_8000,bpe_8000,unigram_4000 총 3가지 tokenizer 조건을 비교하여 여러 시도를 하였다.
        <img width="504" height="471" alt="image" src="https://github.com/user-attachments/assets/c09853f6-a4e5-4dd3-a6da-42ea5b87ffeb" />


- [X]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부

      -> 개인적인 회고는 부족하지만 프로젝트에 대한 간단한 정리와 실험결과에 대한 결론을 작성하였다.
           
        <img width="1459" height="189" alt="image" src="https://github.com/user-attachments/assets/4b312c44-09df-41ad-86d3-555c796819d1" />
        

- [X]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화/모듈화했는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부

     CONFIG로 하이퍼파라마티를 관리하여 동일한 조건에서 실험이 이루어지도록 하였고, tokenizer별 비교 학습을 하나의 fit_tokenizer_option() 함수로 처리하여 동일한 코드를 여러 번 작성하지 않고 실험 옵션만 변경할 수 있도록 구현한 점이 좋았던 것 같습니다.
          <img width="513" height="213" alt="image" src="https://github.com/user-attachments/assets/1070c764-1b65-43fc-9f4b-2564f9b19534" />


    


# 회고(참고 링크 및 코드 개선)
```
# 리뷰어의 회고를 작성합니다.
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
김시온 : 개인적인 회고만 부족한 점을 제외하면 모든 조건을 만족하는 완벽한 코드라고 생각이 든다. 각 과정을 주석을 달아 코드를 이해하기 쉽게 해두었던 점이 좋았고, 여러가지 시도를 해본 점이 좋았던 코드라고 생각이 든다.
