# SkinCare AI

SkinCare AI는 Streamlit으로 만든 피부 이미지 분석 애플리케이션입니다. 사용자는 피부 이미지를 업로드하거나 웹캠으로 촬영해 분석할 수 있습니다. 앱은 선택한 이미지 분류 모델의 예측 상위 결과를 그래프로 보여주고, 질환 안내 엑셀에 일치하는 항목이 있으면 한국어 질환명과 치료 및 피부 관리 정보를 함께 표시합니다.

> 이 앱의 결과는 참고용이며 의료진의 진단을 대체하지 않습니다. 실제 진단과 치료 방법은 피부과 전문의와 상담해야 합니다.

## 주요 기능

- **이미지 분석**: JPG, JPEG, PNG 이미지를 업로드해 피부 질환을 분석합니다.
- **웹캠 분석**: 웹캠으로 촬영한 이미지를 분석합니다.
- **모델 선택**: `CNN_models/` 폴더에 있는 지원 모델 파일 중 사용할 모델을 사이드바에서 선택합니다. 기본 선택은 `YOLO.pt`입니다.
- **예측 결과**: 최대 5개의 예측 클래스와 신뢰도를 그래프로 표시하고, 가장 높은 예측 결과를 강조합니다.
- **질환 정보**: 최고 예측 클래스와 `Skin_Tumor_Disease_Guide.xlsx`의 영어 병명이 일치하면 한국어 병명, 피부과 치료 추천, 추천 피부 관리 정보를 표시합니다.
- **분석 화면**: 어두운 테마의 Streamlit 화면에서 업로드 또는 웹캠 분석 방식을 선택할 수 있습니다.

## 구현 개요

### 애플리케이션 동작

- `app.py`가 Streamlit 앱, 모델 로딩, 이미지 전처리, 예측 및 결과 화면을 담당합니다.
- Ultralytics YOLO 체크포인트는 YOLO 모델로 로드하고, PyTorch `state_dict` 체크포인트는 구조에 따라 ResNet18, MobileNetV3, ConvNeXt-Tiny 또는 EfficientNet-B3 분류기로 로드합니다. 모델 유형은 확장자만이 아니라 체크포인트 내용으로 판별합니다.
- 분류 모델 체크포인트에는 가중치와 `class_names` 또는 `classes` 목록이 들어 있어야 합니다. 분류 이미지는 224×224 크기로 변환하고 ImageNet 평균 및 표준편차로 정규화합니다.
- YOLO 분류 모델은 클래스 확률을 사용하고, 탐지 모델은 상자별 신뢰도를 정렬해 상위 예측을 만듭니다.
- 모델은 `st.cache_resource`로 캐시하고, 질환 안내 엑셀은 `st.cache_data`로 캐시합니다.
- 업로드 및 촬영 이미지는 앱 메모리에서 처리합니다. 이 코드에는 사용자 계정이나 별도의 이미지 저장 기능이 없습니다.

### 모델 및 질환 안내 데이터

- `CNN_models/`에는 앱에서 사용할 `.pt` 또는 `.pth` 가중치 파일을 둡니다. 현재 프로젝트에는 `YOLO.pt`, `resnet18.pth`, `SkinCare_MobileNetV3_Improved_Model.pth`, `EfficientNet-B3.pth`가 포함되어 있습니다.
- YOLO 모델 클래스명 또는 분류 체크포인트의 클래스명이 엑셀의 `영어병명` 값과 일치해야 치료 안내를 찾을 수 있습니다. 비교할 때 대소문자와 공백은 정규화됩니다.
- `Skin_Tumor_Disease_Guide.xlsx`의 `질환 정보` 시트에서 5행을 열 제목으로 읽으며 `영어병명`, `한국병명`, `피부과치료추천`, `추천화장품` 열을 사용합니다.
- `CNN_models_code/`에는 YOLO, ResNet18, MobileNetV3, EfficientNet-B3, ConvNeXt-Tiny 학습 및 실험 노트북이 있습니다. 현재 앱은 위 네 가지 PyTorch 분류 구조와 Ultralytics YOLO 체크포인트를 지원합니다. ConvNeXt와 EfficientNet 체크포인트는 학습 때 사용한 알파벳 순 15개 클래스 이름을 앱에서 연결합니다.

## 프로젝트 구조

```text
.
├── app.py                                  # Streamlit 앱과 추론 및 결과 화면
├── requirements.txt                        # Python 패키지 의존성
├── Skin_Tumor_Disease_Guide.xlsx           # 질환별 안내 데이터
├── CNN_models/                             # 앱에서 사용하는 .pt/.pth 가중치
├── CNN_models_code/                        # 모델 학습 및 실험 노트북
│   ├── SkinCare_YOLO_Model.ipynb
│   ├── SkinCare_ResNet18_Model.ipynb
│   ├── SkinCare_MobileNetV3_Model.ipynb
│   └── SkinCare_EfficientNet-B3.ipynb
└── sc_page.ipynb                           # 프로젝트 분석/실험 노트북
```

## 실행 방법

### 사전 준비

- Python 3.10 이상을 권장합니다.
- 저장소 루트에 `app.py`, `Skin_Tumor_Disease_Guide.xlsx`가 있어야 합니다.
- 추론에 사용할 `.pt` 또는 `.pth` 파일을 `CNN_models/` 폴더에 둡니다. 앱 실행 시 폴더가 없으면 생성되지만, 지원 가중치 파일이 없으면 앱은 오류를 표시하고 종료합니다.

### Windows PowerShell

저장소 루트에서 가상 환경을 만들고 의존성을 설치한 다음 앱을 실행합니다.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit이 출력하는 로컬 주소를 브라우저에서 열어 앱을 사용합니다. 기본 주소는 <http://localhost:8501>입니다. 첫 실행 시 딥러닝 패키지 설치와 모델 로딩에 시간이 걸릴 수 있습니다.

## 사용 방법

1. 사이드바에서 `이미지 분석` 또는 `웹캠 분석`을 선택합니다.
2. 사용할 모델을 선택합니다.
3. 이미지 분석에서는 JPG, JPEG, PNG 파일을 업로드합니다. 웹캠 분석에서는 촬영 버튼을 눌러 이미지를 입력합니다.
4. 예측 그래프와 최고 예측 결과를 확인합니다. 클래스명이 엑셀 데이터와 일치하면 추가 질환 안내도 표시됩니다.

## 문제 해결

- **모델 파일이 없다는 오류**: `CNN_models/`에 `.pt` 또는 `.pth` 파일이 있는지 확인합니다.
- **질환 안내 엑셀을 불러오지 못한다는 오류**: 엑셀 파일명이 `Skin_Tumor_Disease_Guide.xlsx`인지, `질환 정보` 시트와 필요한 열이 있는지 확인합니다.
- **분류 모델 클래스명 오류**: 체크포인트에 `class_names` 또는 `classes` 목록이 포함되어 있는지 확인합니다.
- **치료 정보를 찾지 못함**: 모델의 클래스명이 엑셀의 `영어병명`과 대응하는지 확인합니다.
- **예측 결과가 비어 있음**: 모델이 분류 확률 또는 탐지 상자를 반환하는지, 입력 이미지가 정상적으로 열리는지 확인합니다.

## 참고

- 의존성 버전은 `requirements.txt`에서 관리합니다.
- 학습용 데이터와 모델 학습 절차는 `CNN_models_code/`의 노트북을 참고하세요.
- 본 프로젝트는 교육 및 참고용 피부 이미지 분석 도구이며 의학적 진단이나 치료를 제공하지 않습니다.
