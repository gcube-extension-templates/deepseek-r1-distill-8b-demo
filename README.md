# 🧠 DeepSeek-R1-Distill-Llama-8B Demo

> 💡 **추론 특화 모델의 사고 과정을 직접 확인해보고 싶은 분께 추천하는 프로젝트입니다.**  
> DeepSeek R1의 단계적 사고(`<think>`) 과정을 **Open WebUI** 기반 ChatGPT 같은 웹 인터페이스로 직접 확인해볼 수 있습니다.

---

## 📌 모델 소개

**DeepSeek-R1-Distill-Llama-8B**는 중국 AI 스타트업 DeepSeek가 2025년 1월 공개한 추론 특화 오픈소스 모델입니다.

### 역사와 배경
2025년 초 DeepSeek는 OpenAI의 o1과 유사한 추론 능력을 가진 R1 모델을 공개했습니다. 놀라운 점은 OpenAI 대비 훨씬 적은 비용으로 개발되었다는 것으로, 공개 당시 AI 업계에 큰 충격을 주었습니다. 특히 R1-Distill 시리즈는 대형 모델의 추론 능력을 소형 모델에 증류(Distill)한 것으로, 8B라는 작은 크기에도 불구하고 수학·논리 문제에서 뛰어난 성능을 보입니다.

### 장단점

| 장점 | 단점 |
|------|------|
| 수학, 논리, 코딩 문제에서 뛰어난 성능 | 일반 대화에서 범용 모델보다 느릴 수 있음 |
| `<think>` 태그로 추론 과정 투명하게 공개 | `<think>` 과정이 길어 응답 시간이 상대적으로 김 |
| MIT 라이선스로 상업적 이용 가능 | 한국어 성능이 영어 대비 다소 낮음 |
| Llama 3.1 아키텍처 기반으로 생태계 호환 | |

### 동작 원리
Llama 3.1 8B 아키텍처를 베이스로, DeepSeek R1 대형 모델의 추론 능력을 증류(Knowledge Distillation)한 모델입니다. GRPO(Group Relative Policy Optimization) 강화학습으로 단계적 사고 능력을 훈련했으며, 최종 답변 전 `<think>` 태그 안에서 문제를 분해하고 검증하는 과정을 거칩니다. Open WebUI에서는 이 추론 과정을 접을 수 있는 형태로 확인할 수 있습니다.

| 항목 | 내용 |
|------|------|
| 개발사 | DeepSeek AI |
| 파라미터 수 | 8B (80억) |
| 라이선스 | MIT License (무료, 상업적 이용 가능) |
| Ollama | [ollama.com/library/deepseek-r1](https://ollama.com/library/deepseek-r1) |

---

## ⚙️ 시스템 요구사항

| 항목 | 최소 사양 | 권장 사양 |
|------|-----------|-----------|
| GPU | NVIDIA RTX 4060 (8GB VRAM) | NVIDIA RTX 4080 / 4090 |
| GPU 아키텍처 | Ada Lovelace (RTX 40 시리즈) | Ada Lovelace (RTX 40 시리즈) |
| CUDA | 12.1 이상 | 12.4 이상 |
| RAM | 16GB | 32GB |
| 저장공간 | 10GB 이상 | 20GB 이상 |
| Python | 3.10 이상 | 3.11 |

> ✅ Ollama가 자동으로 최적 양자화 포맷(GGUF)을 선택하므로 RTX 40 시리즈 전 라인업(RTX 4060 8GB~)에서 실행 가능합니다.  
> ✅ MIT 라이선스로 HuggingFace 계정 및 토큰이 **필요하지 않습니다.**

---

## 🚀 빠른 시작

### 1. 저장소 클론

> ⚠️ 워크로드 배포 직후에는 git/curl이 아직 설치 중일 수 있습니다.  
> `git: command not found` 오류가 발생하면 잠시 후 다시 시도하거나 아래 명령어로 수동 설치하세요.
> ```bash
> apt-get update && apt-get install -y git curl
> ```

```bash
git clone https://github.com/gcube-extension-templates/deepseek-r1-distill-8b-demo.git
cd deepseek-r1-distill-8b-demo
```

### 2. Setup 실행 (Ollama + Open WebUI 설치 + 모델 다운로드)

```bash
bash setup.sh
```

> ⏳ Ollama, Open WebUI 설치 및 모델 다운로드가 자동으로 진행됩니다.

### 3. 서비스 시작

```bash
bash start.sh
```

> 터미널에 **"Open WebUI 준비 완료!"** 메시지가 출력된 후 브라우저에서 접속하세요.

### 4. 브라우저에서 접속

GCUBE 워크로드의 **서비스 URL (포트 8080)** 으로 접속하면 채팅 인터페이스가 열립니다.

---

## 📁 프로젝트 구조

```
deepseek-r1-distill-8b-demo/
├── README.md        # 프로젝트 설명 (현재 파일)
├── setup.sh         # Ollama + Open WebUI 설치 + 모델 다운로드
└── start.sh         # 서비스 시작 스크립트
```

---

## ❓ 자주 묻는 질문

**Q. 처음 실행 시 시간이 오래 걸려요.**  
A. setup.sh 실행 중 Ollama, Open WebUI 설치 및 모델 다운로드가 진행됩니다. 인터넷 속도에 따라 수~수십 분이 소요될 수 있으며, 다음 실행부터는 `bash start.sh`만 실행하면 됩니다.

**Q. `<think>` 태그가 뭔가요?**  
A. DeepSeek R1 모델이 최종 답변 전 단계적으로 추론하는 과정을 표시하는 태그입니다. Open WebUI에서는 접을 수 있는 형태로 보여줍니다. 수학이나 논리 문제를 입력하면 추론 과정을 확인할 수 있습니다.

**Q. 브라우저에서 접속이 안 돼요.**  
A. `bash start.sh` 실행 후 터미널에 "Open WebUI 준비 완료!" 메시지가 출력됐는지 확인해주세요. GCUBE 워크로드 설정에서 포트 8080이 열려 있는지도 확인하세요.

---

## 📜 라이선스

이 프로젝트의 코드 및 모델 가중치 모두 MIT License를 따릅니다.