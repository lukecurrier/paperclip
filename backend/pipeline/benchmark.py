import os
import csv
import torch
import argparse
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

bert_model = AutoModel.from_pretrained("bert-base-uncased").to(device)
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def get_similarity(pred, reference):
    def embed(text):
        tokens = bert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(device)

        with torch.no_grad():
            out = bert_model(**tokens).last_hidden_state[:, 0, :]

        return out.cpu().numpy()

    emb1 = embed(pred)
    emb2 = embed(reference)

    return cosine_similarity(emb1, emb2)[0][0]


def run_openai(model_name, text):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You summarize documents clearly and concisely."},
            {"role": "user", "content": f"Summarize:\n\n{text}"}
        ],
        temperature=0
    )

    return response.choices[0].message.content


def load_local_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path).to(device)
    model.eval()
    return model, tokenizer


def run_local(model, tokenizer, text):
    model.eval()

    MAX_LEN = min(1024, getattr(model.config, "n_positions", 1024))

    # STEP 1: tokenize with HARD truncation
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # STEP 2: FORCE CLIP (this is the real safety net)
    input_ids = input_ids[:, -MAX_LEN:]
    attention_mask = attention_mask[:, -MAX_LEN:]

    # STEP 3: DEBUG (DO NOT SKIP THIS)
    print("DEBUG seq_len =", input_ids.shape[1], "max =", MAX_LEN)

    # STEP 4: ensure device consistency
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    # STEP 5: generate safely
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_benchmark(mode, model, tokenizer, csv_file, text_key, summary_key):
    print(f"\nRunning benchmark: {csv_file}")

    reader = csv.DictReader(open(csv_file, encoding="utf-8"))

    total = 0
    score_sum = 0

    for index, row in enumerate(reader):
        if index >= 30:
            break

        text = row[text_key][:8000]
        reference = row[summary_key]

        # model inference
        if mode == "openai":
            pred = run_openai(model, text)
        else:
            pred = run_local(model, tokenizer, text)

        score = get_similarity(pred, reference)

        score_sum += score
        total += 1

        if total % 10 == 0:
            print(f"Processed {total} samples...")

    print(f"\nFinal Score for {csv_file}: {score_sum / total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["openai", "local"], required=True)

    args = parser.parse_args()

    if args.mode == "openai":
        model = "gpt-4o-mini"
        tokenizer = None

    else:
        model, tokenizer = load_local_model("backend/pipeline/finetuning/cpu_output/final_model")

    run_benchmark(
        args.mode,
        model,
        tokenizer,
        "backend/pipeline/data/benchmark_files/usb.csv",
        "input_lines",
        "output_lines"
    )