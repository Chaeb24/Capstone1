# Capstone ['인공지능을 이용한 안개 제거연구']
캡스톤 연구에 사용한 모든 코드 첨부

## 추론 모델 코드
HAZE4K : NHHAZE (1:10) 학습 모델에 각각 ITS / OTS 추가로 10만 iteration 파인튜닝

## SwinIR 코드
HAZE4K : NHHAZE (1:10) 데이터로 학습, 20만 iteration  
추가로 SwinIR은 charbonnier, l2 기법 적용

## GUI 데모 코드
FFA-NET 모델 선택, SwinIR 모델 선택 후  
안개가 낀 이미지를 넣으면 왼쪽은 FFA-NET 적용, 오른쪽은 SwinIR 적용된 clear image  
확인 가능

## 사용 지표
PSNR(Peak Signal-to-Noise Ratio)  
원본과 비교했을 때 얼마만큼 개선되었는지를 보여줌. 30dB이상이면 양호하다고 평가함.

SSIM(Structural Similarity Index)  
사람이 보았을 때 얼마나 원본과 유사한지를 판단함.  
밝기, 명암비, 구조 정보까지 반영, 0~1사이로 수치를 표현하며 높을수록 좋음.
