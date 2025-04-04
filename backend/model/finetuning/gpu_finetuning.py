
import wandb
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import argparse

torch.cuda.empty_cache()
bert_model = AutoModel.from_pretrained("bert-base-uncased", config={"use_flash_attention": False})
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
device = "cuda"
bert_model = bert_model.to(device)


def format_item(item):
    question = item["text"].strip()
    answer = item["summary"]
    return {"content": f"# Text\n\n{question}\n\n# Summary\n\n{answer}"}


def main():
    wandb.init(project="llm_systems")

    train_data = load_dataset("FiscalNote/billsum", split="train"
    ).select(range(2000)).map(format_item)
    test_data = load_dataset("FiscalNote/billsum", split="test"
    ).select(range(2000)).map(format_item)
    

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        modules_to_save=["lm_head", "embed_token"],
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        dataset_text_field="content",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_seq_length=1028,
        output_dir="output",
        learning_rate=3e-05,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        bf16=False,
        eval_strategy="steps",
        eval_steps=10,
        logging_steps=1,
        run_name="gsm8k-peft-sft",
    )

    trainer = SFTTrainer(
        "meta-llama/Llama-3.2-1B",
        train_dataset=train_data,
        eval_dataset=test_data,
        args=sft_config,
        peft_config=peft_config,
    )

    trainer.train()
    
def run_benchmark(model, tokenizer, benchmark, text_key, summary_key):
    prompt = f"""# Text\n\n{benchmark[text_key]}\n\n# Summary\n\n"""
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    outputs = model.generate(
        inputs["input_ids"],
        pad_token_id = tokenizer.eos_token_id,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.2,
        attention_mask=inputs["attention_mask"],
        top_p=None)
    
    resp_content = tokenizer.decode(outputs[0])
    
    #print(resp_content)
    #print(benchmark['summary'])
    
    input_ids1 = torch.tensor(bert_tokenizer.encode(resp_content, truncation=True)).unsqueeze(0).to(device)
    input_ids2 = torch.tensor(bert_tokenizer.encode(benchmark[summary_key], truncation=True)).unsqueeze(0).to(device)

    # Obtain the BERT embeddings
    with torch.no_grad():
        outputs1 = bert_model(input_ids1)
        outputs2 = bert_model(input_ids2)
        embeddings1 = outputs1.last_hidden_state[:, 0, :].cpu()
        embeddings2 = outputs2.last_hidden_state[:, 0, :].cpu()

    similarity_score = cosine_similarity(embeddings1, embeddings2)
    #print("Similarity Score:", similarity_score)
    return similarity_score

def evaluate_finetuned_benchmarks(csv_name, text_key, summary_key):
    model = AutoModelForCausalLM.from_pretrained("output/checkpoint-250", config={"use_flash_attention": False}).to(device)
    tokenizer = AutoTokenizer.from_pretrained("output/checkpoint-250")

    total_benchmarks = 0
    sim_sum = 0
    
    df = pd.read_csv(csv_name)
    for i, row in df.iterrows():
        if i >= 2000:
            break
        total_benchmarks = total_benchmarks + 1
        sim_sum = sim_sum + run_benchmark(model, tokenizer, row, text_key, summary_key)
    print("Total Accuracy: ", sim_sum/total_benchmarks)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation or main function.")
    parser.add_argument(
        "--mode", choices=["finetune", "benchmark"], required=True,
        help="Choose whether to run benchmark evaluation or finetune."
    )
    
    args = parser.parse_args()

    if args.mode == "benchmark":
        evaluate_finetuned_benchmarks('data/benchmarks/usb.csv', 'input_lines', 'output_lines')
        evaluate_finetuned_benchmarks('data/benchmarks/filtered_articles.csv', 'text', 'summary')
    elif args.mode == "finetune":
        main()
