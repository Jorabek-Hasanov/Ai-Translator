import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def translate_en_to_uz(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    # Move inputs to GPU if available
    if torch.cuda.is_available():
        model.to("cuda")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
    outputs = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

if __name__ == "__main__":
    print("--- AI Tekshiruv ---")
    print(f"PyTorch versiyasi: {torch.__version__}")
    print(f"Transformers versiyasi: {transformers.__version__}")

    if torch.cuda.is_available():
        print("GPU (Video karta) aniqlandi va ishlamoqda!")
    else:
        print("GPU topilmadi, CPU orqali hisoblanadi.")

    print("\n--- Tarjima modeli yuklanmoqda... ---")
    try:
        model_name = "booba-uz/english-uzbek-translation_v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        test_text = "Hello world! I am learning artificial intelligence."
        print(f"\nOriginal: {test_text}")
        print(f"Tarjima: {translate_en_to_uz(test_text, model, tokenizer)}")

    except Exception as e:
        print(f"Xatolik yuz berdi: {e}")