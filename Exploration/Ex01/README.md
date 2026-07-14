# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 조영근
- 리뷰어 : 김나연


# PRT(Peer Review Template)
- [X]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 필요한 부분에 시각화가 적절히 이루어졌다.
        - <img width="782" height="688" alt="image" src="https://github.com/user-attachments/assets/d63b9b74-664c-452c-90b8-caf69956ec27" />
        - <img width="1523" height="792" alt="image" src="https://github.com/user-attachments/assets/81939f2f-6540-4e3f-955a-d77738dfa7d7" />
        - <img width="1817" height="680" alt="image" src="https://github.com/user-attachments/assets/7ef795b9-7693-4d18-878e-fc0ec1363dca" />
    - 프로젝트 1의 회귀모델 예측정확도가 MSE 손실함수값 3000 이하를 달성하였다.
        - <img width="847" height="301" alt="image" src="https://github.com/user-attachments/assets/bfa8f0ba-68b7-4e2d-9a33-9ecd984779b3" />
    - 프로젝트 2의 회귀모델 예측정확도가 RMSE 값 150 이하를 달성하였다.
        - <img width="832" height="615" alt="image" src="https://github.com/user-attachments/assets/73fc8357-2df6-44c9-9811-b85c2bb312db" />
    
- [X]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - <img width="528" height="187" alt="image" src="https://github.com/user-attachments/assets/690f8038-a866-4adf-a725-1c0d83a057eb" />
        - 여러 학습률을 비교하고 실험한 결과로 최적의 학습률을 찾아낸 점이 핵심적이다.
        - 주석에서 학습률을 0.001부터 점진적으로 조정하며 실험한 결과 0.5가 가장 적합했다고 설명한 점이 이해하기 쉽다.
        
- [X]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    - <img width="712" height="377" alt="image" src="https://github.com/user-attachments/assets/94a5906b-ec18-4ad7-add7-ad2a8b334f9e" />
        - 파생변수로 'weekday'를 새롭게 생성했다.
        - 타겟 y를 로그 변환을 통해 스케일링 하였다.
        - RandomForestRegressor 모델을 사용하였다.
        
- [ ]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
        
- [X]  **5. 코드가 간결하고 효율적인가요?**
    - <img width="611" height="676" alt="image" src="https://github.com/user-attachments/assets/d1e06b30-bb17-4612-9459-b5b4e66907af" />
        - 전체 과정을 model 함수, MSE 함수, loss 함수, gradient 함수로 간결하게 나누어 작성하였다.


# 회고(참고 링크 및 코드 개선)
    - 학습률 조정 시 직접 학습률을 대입해보는 것보다 for 문을 사용했다면 실험 결과를 시각적으로 표현할 수도 있고 번거로움을 덜었을 것 같다.
    - 새로운 모델을 사용하거나 학습률 조정을 통해 최소한의 loss 값을 찾아가는 과정이 인상 깊었다.
    
```
learning_rates = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5]

final_losses = []

for lr in learning_rates:
    # 가중치 초기화
    W = 3.0
    b = 1.0

    losses = []

    # 모델 학습
    for i in range(1, 1001):
        dW, db = gradient(X_train, W, b, y_train)
        W -= lr * dW
        b -= lr * db

        L = loss(X_train, W, b, y_train)
        losses.append(L)

    final_losses.append(losses[-1])

    print(f"Learning Rate: {lr:<5}  Final Loss: {losses[-1]:.4f}")

# 최적의 학습률 찾기
best_idx = np.argmin(final_losses)
best_lr = learning_rates[best_idx]

print(f"\n최적의 학습률: {best_lr}")

import matplotlib.pyplot as plt

plt.figure(figsize=(7,4))
plt.plot(learning_rates, final_losses, marker='o')
plt.title("Learning Rate vs Final Loss")
plt.xlabel("Learning Rate")
plt.ylabel("Final Loss")
plt.grid(True)
plt.show()
```
