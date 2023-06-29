# PytorchLightning_framework
### <Fashion MNIST 학습 예제 실습>

![Alt text](image-1.png)

- 코드의 추상화를 통해, 프레임워크를 넘어 하나의 코드 스타일로 작성하기 위함
- **Lightning Module** : pl.LightningModule 클래스를 상속받아 모델의 구조, 데이터 전처리, 손실함수 등의 설정을 통해 "모델을 초기화"
- **Trainer** : 모델의 학습 epoch이나 batch, 로그 생성까지 "모델의 학습을 담당"

