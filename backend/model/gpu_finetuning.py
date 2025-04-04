"""
The following script will work on henry.khoury.northeastern.edu, which
has an RTX 6000 GPU that was purchased in 2020. This should serve as a
template for fine-tuning any 1B parameter model on this machine. The
specific things to notice are:

- Don't try to go higher than 3B parameters.
- Use a *parameter-efficient fine-tuning* method. This script uses LoRA.
- Use a relatively small batch size (per_device_train_batch_size).
- Use a relatively small maximum sequence length (max_seq_length).

You can see the training log here:

https://wandb.ai/nuprl/llm_systems/runs/gghys2ro

You should be able to create your own Wandb account and reproduce the log
above. It takes less than 10 minutes to run.
"""

import wandb
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


torch.cuda.empty_cache()
bert_model = AutoModel.from_pretrained("bert-base-uncased", config={"use_flash_attention": False})
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
device = "cuda"
bert_model = bert_model.to(device)


def format_item(item):
    question = item["text"].strip()
    answer = item["summary"]
    return {"content": f"# Text\n\n{question}\n\n# Summary\n\n{answer}"}


# def format_item(item):
#     question = item["question"].strip()
#     answer = item["answer"]
#     return {"content": f"# Question\n\n{question}\n\n# Answer\n\n{answer}"}

def main():
    wandb.init(project="llm_systems")

    train_data = load_dataset("FiscalNote/billsum", split="train"
    ).map(format_item).select(range(1000))
    test_data = load_dataset("FiscalNote/billsum", split="test"
    ).map(format_item).select(range(1000))
    
    # train_data = load_dataset(
    #     "nuprl/engineering-llm-systems", "gsm8k", split="train"
    # ).map(format_item)
    # test_data = load_dataset(
    #     "nuprl/engineering-llm-systems", "math_word_problems", split="test"
    # ).map(format_item)

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
    
def run_benchmark(model, tokenizer, benchmark):
    prompt = f"""# Text\n\n{benchmark['text']}\n\n# Summary\n\n"""
    
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
    
    tokenized_resp = bert_tokenizer.tokenize(resp_content)
    tokenized_benchmark = bert_tokenizer.tokenize(benchmark['summary'])
    
    #print(resp_content)
    #print(benchmark['summary'])
    
    input_ids1 = torch.tensor(bert_tokenizer.encode(resp_content, truncation=True, max_length=512)).unsqueeze(0).to(device)
    input_ids2 = torch.tensor(bert_tokenizer.encode(benchmark['summary'], truncation=True, max_length=512)).unsqueeze(0).to(device)

    # Obtain the BERT embeddings
    with torch.no_grad():
        outputs1 = bert_model(input_ids1)
        outputs2 = bert_model(input_ids2)
        embeddings1 = outputs1.last_hidden_state[:, 0, :].cpu()
        embeddings2 = outputs2.last_hidden_state[:, 0, :].cpu()

    similarity_score = cosine_similarity(embeddings1, embeddings2)
    print("Similarity Score:", similarity_score)
    return similarity_score

def evaluate_benchmark():
    #grab from wherever it is saved
    model = AutoModelForCausalLM.from_pretrained("output/checkpoint-125", config={"use_flash_attention": False}).to(device)
    tokenizer = AutoTokenizer.from_pretrained("output/checkpoint-125")
    
    total_benchmarks = 0
    sim_sum = 0
    
    df = pd.read_csv('benchmark.csv')
    for i, row in df.iterrows():
        if i >= 20:
            break
        total_benchmarks = total_benchmarks + 1
        sim_sum = sim_sum + run_benchmark(model, tokenizer, row)
    print("Total Accuracy: ", sim_sum/total_benchmarks)
        
    

if __name__ == "__main__":
    evaluate_benchmark()
    #main()
