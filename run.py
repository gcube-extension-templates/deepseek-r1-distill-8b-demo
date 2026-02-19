"""
DeepSeek-R1-Distill-Llama-8B — Reasoning Model Demo
=====================================================
DeepSeek R1의 추론 능력을 Llama 8B에 증류한 모델을 실행합니다.
<think> 태그를 파싱하여 추론 과정과 최종 답변을 분리 출력합니다.

추천 질문 유형: 수학 문제, 코딩 문제, 논리 추론 문제
"""

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# =============================================
# 설정 (원하는 대로 수정 가능)
# =============================================

# 모델 경로 (setup.sh 실행 후 로컬 경로 사용)
# setup.sh를 실행하지 않은 경우 Hugging Face Hub에서 자동 다운로드
MODEL_ID = "/workspace/models/deepseek-r1-distill-llama-8b"
FALLBACK_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# 추론 질문 (수학/코딩/논리 문제를 입력하면 <think> 과정이 잘 나타납니다)
USER_PROMPT = "피보나치 수열의 10번째 수를 단계별로 구하세요."

# 생성할 최대 토큰 수 (추론 과정 포함으로 넉넉히 설정)
MAX_NEW_TOKENS = 2048

# Temperature (DeepSeek 공식 권고: 0.5 ~ 0.7)
TEMPERATURE = 0.6

# 4-bit 양자화 사용 여부
# RTX 40 시리즈 전 라인업(최소 RTX 4060 8GB) 호환을 위해 기본값 True
# VRAM 16GB 이상 환경에서 False로 변경 시 더 높은 품질로 실행 가능
USE_4BIT = True

# =============================================


def load_model(model_id: str, use_4bit: bool):
    """모델과 토크나이저를 로드합니다."""
    import os
    # 로컬 경로에 모델이 없으면 Hugging Face Hub에서 다운로드
    if not os.path.exists(model_id):
        print(f"⚠️  로컬 모델 경로({model_id})를 찾을 수 없습니다.")
        print(f"   Hugging Face Hub에서 다운로드합니다: {FALLBACK_MODEL_ID}")
        model_id = FALLBACK_MODEL_ID
    print(f"모델 로딩 중... ({model_id})")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }

    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print("  ℹ️  4-bit 양자화 모드로 실행합니다 (VRAM 절약)")

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    print("✅ 모델 로딩 완료!\n")
    return model, tokenizer


def parse_response(full_text: str) -> tuple[str, str]:
    """
    모델 출력에서 <think>...</think> 추론 과정과 최종 답변을 분리합니다.

    Returns:
        (think_content, answer_content) 튜플
    """
    think_match = re.search(r"<think>(.*?)</think>", full_text, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else ""

    # </think> 이후의 텍스트를 최종 답변으로 처리
    if "</think>" in full_text:
        answer_content = full_text.split("</think>", 1)[1].strip()
    else:
        answer_content = full_text.strip()

    return think_content, answer_content


def generate_response(
    model, tokenizer, user_prompt: str, max_new_tokens: int, temperature: float
) -> str:
    """
    모델에 질문을 전달하고 전체 응답(추론 과정 포함)을 반환합니다.

    Note:
        DeepSeek 공식 권고사항:
        - System prompt 없이 User prompt만 사용
        - Temperature: 0.5 ~ 0.7 (기본값 0.6)
    """
    # DeepSeek-R1-Distill 모델은 system prompt 없이 사용 권장
    messages = [
        {"role": "user", "content": user_prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 입력 토큰 제외하고 생성된 부분만 디코딩
    generated_ids = output_ids[0][input_ids.shape[1]:]
    full_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return full_output


def main():
    print("=" * 50)
    print("  🧠 DeepSeek-R1-Distill-Llama-8B — Demo")
    print("=" * 50)

    # GPU 상태 확인
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🖥️  GPU: {gpu_name} ({vram:.1f}GB VRAM)")
    else:
        print("⚠️  GPU를 찾을 수 없습니다. CPU로 실행합니다 (매우 느릴 수 있음)")

    print()

    # 모델 로드
    model, tokenizer = load_model(MODEL_ID, USE_4BIT)

    # 응답 생성
    print(f"[질문] {USER_PROMPT}\n")

    full_output = generate_response(
        model, tokenizer, USER_PROMPT, MAX_NEW_TOKENS, TEMPERATURE
    )

    # 추론 과정과 최종 답변 분리
    think_content, answer_content = parse_response(full_output)

    # 추론 과정 출력
    if think_content:
        print("--- 추론 과정 (Thinking) ---")
        print(f"<think>\n{think_content}\n</think>")
        print()

    # 최종 답변 출력
    print("--- 최종 답변 ---")
    print(answer_content)
    print("=" * 50)
    print("\n✅ 완료! run.py의 USER_PROMPT를 수정해서 다른 질문을 해보세요.")
    print("   수학, 코딩, 논리 문제를 입력하면 추론 과정이 더 잘 나타납니다.")


if __name__ == "__main__":
    main()