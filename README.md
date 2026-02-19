# 🧠 DeepSeek-R1-Distill-Llama-8B — Reasoning Model Demo

> 💡 **AI 모델에 어느 정도 익숙한 개발자에게 추천하는 프로젝트입니다.**  
> 단순한 텍스트 생성을 넘어, 모델이 **추론 과정(`<think>`)을 직접 출력**하는 Reasoning 모델을 경험해볼 수 있습니다.

---

## 📌 모델 소개

**DeepSeek-R1-Distill-Llama-8B**는 중국 AI 스타트업 DeepSeek이 2025년 1월 공개한  
`DeepSeek-R1` 추론 모델의 지식을 Llama 8B 아키텍처에 **증류(Distillation)** 한 경량 모델입니다.

DeepSeek-R1은 OpenAI의 o1 모델과 맞먹는 수학/코딩/논리 추론 능력을 보여주며  
**오픈소스 AI의 패러다임을 바꾼 모델**로 평가받고 있습니다.  
이 프로젝트는 그 능력을 RTX 40 시리즈 단일 GPU에서 실행할 수 있는 8B 경량 버전을 사용합니다.

| 항목 | 내용 |
|------|------|
| 개발사 | DeepSeek AI |
| 베이스 모델 | Meta Llama 3.1 8B |
| 학습 방식 | Knowledge Distillation from DeepSeek-R1 |
| 라이선스 | MIT License |
| Hugging Face | [deepseek-ai/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) |

---

## 🔍 Reasoning 모델이란?

일반 LLM은 질문에 바로 답변을 생성합니다.  
Reasoning 모델은 답변 전에 **`<think>` 태그 내부에서 단계별 추론 과정**을 먼저 수행합니다.

```
[입력] 1부터 10까지의 합은?

[모델 출력]
<think>
1부터 10까지 더하면... 1+2=3, 3+3=6, 6+4=10, 10+5=15...
등차수열 공식 n*(n+1)/2를 쓰면 10*11/2 = 55
</think>

1부터 10까지의 합은 **55**입니다.
```

이 방식으로 수학, 코딩, 논리 추론 문제에서 훨씬 높은 정확도를 보입니다.

---

## ⚙️ 시스템 요구사항

| 항목 | 최소 사양 | 권장 사양 |
|------|-----------|-----------|
| GPU | NVIDIA RTX 4060 (8GB VRAM) | NVIDIA RTX 4080 / 4090 |
| GPU 아키텍처 | Ada Lovelace (RTX 40 시리즈) | Ada Lovelace (RTX 40 시리즈) |
| CUDA | 12.1 이상 | 12.4 이상 |
| RAM | 16GB | 32GB |
| 저장공간 | 20GB (모델 다운로드 포함) | 30GB 이상 |
| Python | 3.10 이상 | 3.11 |

> ✅ MIT 라이선스로 **Hugging Face 계정 없이** 모델 다운로드가 가능합니다.

---

## 🚀 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/your-org/deepseek-r1-distill-8b-demo.git
cd deepseek-r1-distill-8b-demo
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. Setup 실행 (패키지 설치 + 모델 다운로드 한번에)

```bash
bash setup.sh
```

> ✅ MIT 라이선스 모델로 Hugging Face 로그인 없이 바로 다운로드됩니다.

### 4. 모델 실행

```bash
python run.py
```

### 4. 실행 결과 예시

```
==============================================
  🧠 DeepSeek-R1-Distill-Llama-8B — Demo
==============================================
🖥️  GPU: NVIDIA GeForce RTX 4090 (24.0GB VRAM)
모델 로딩 중...
✅ 모델 로딩 완료!

[질문] 피보나치 수열의 10번째 수를 단계별로 구하세요.

--- 추론 과정 (Thinking) ---
<think>
피보나치 수열은 F(1)=1, F(2)=1로 시작하고
F(n) = F(n-1) + F(n-2)로 정의됩니다.
F(3) = 1+1 = 2
F(4) = 2+1 = 3
F(5) = 3+2 = 5
...
F(10) = 34+21 = 55
</think>

--- 최종 답변 ---
피보나치 수열의 10번째 수는 **55**입니다.

단계별 계산:
F(1)=1, F(2)=1, F(3)=2, F(4)=3, F(5)=5,
F(6)=8, F(7)=13, F(8)=21, F(9)=34, F(10)=55
==============================================
```

---

## 📁 프로젝트 구조

```
deepseek-r1-distill-8b-demo/
├── README.md          # 프로젝트 설명 (현재 파일)
├── requirements.txt   # 필요한 Python 패키지 목록
├── setup.sh           # 패키지 설치 + 모델 다운로드 자동화 스크립트
└── run.py             # 모델 실행 스크립트 (think 파싱 포함)
```

---

## 🛠️ 고급 설정

`run.py` 상단의 설정 변수를 수정해 동작을 조절할 수 있습니다.

```python
# 추론에 사용할 질문 (수학, 코딩, 논리 문제에서 가장 효과적)
USER_PROMPT = "피보나치 수열의 10번째 수를 단계별로 구하세요."

# 생성 토큰 수 (추론 과정 포함이라 넉넉히 설정 권장)
MAX_NEW_TOKENS = 2048

# Temperature (0.5~0.7 권장, DeepSeek 공식 권고값)
TEMPERATURE = 0.6

# 4-bit 양자화 (VRAM 8GB 환경에서 True로 설정)
USE_4BIT = False
```

> 📌 **DeepSeek 공식 권고:** Temperature는 0.5~0.7 범위를 사용하세요.  
> System prompt는 사용하지 않는 것을 권장합니다.

---

## ❓ 자주 묻는 질문

**Q. `<think>` 태그가 출력되지 않아요.**  
A. 모델이 추론 과정을 생략한 경우입니다. 수학/코딩/논리 문제처럼 단계적 사고가 필요한 질문을 사용하면 추론 과정이 잘 나타납니다.

**Q. 일반 Llama 3.1과 어떤 차이가 있나요?**  
A. 이 모델은 DeepSeek-R1의 추론 능력을 증류받아 수학/논리 문제에서 훨씬 높은 정확도를 보입니다. 단, 일반 대화보다는 추론이 필요한 질문에 적합합니다.

**Q. CUDA out of memory 오류가 발생해요.**  
A. `run.py`의 `USE_4BIT = True`로 설정하면 VRAM 사용량을 약 절반으로 줄일 수 있습니다.

---

## 📜 라이선스

이 프로젝트의 코드는 MIT License를 따릅니다.  
모델 가중치도 [MIT License](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)를 따르며 상업적 이용이 가능합니다.