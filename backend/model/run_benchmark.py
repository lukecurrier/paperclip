import os
from dotenv import load_dotenv
import torch
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import csv
import argparse

load_dotenv()

bert_model = AutoModel.from_pretrained("bert-base-uncased")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
def run_benchmark(model_name, benchmark, text_key, summary_key):
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL")
    
    client = OpenAI(api_key=api_key, base_url=base_url)

    SYSTEM_PROMPT = """You are a helpful assistant that summarizes scientific papers."""
    
    USER_PROMPT = f"""Please briefly summarize the following markdown content:
    
    {benchmark[text_key]}

    Summary:"""
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ],
        model=model_name,
        temperature=0,
        max_tokens=300
    )    
    resp_content = response.choices[0].message.content
    
    input_ids1 = torch.tensor(bert_tokenizer.encode(resp_content, truncation=True, max_length=512)).unsqueeze(0)
    input_ids2 = torch.tensor(bert_tokenizer.encode(benchmark[summary_key], truncation=True, max_length=512)).unsqueeze(0)

    # Obtain the BERT embeddings
    with torch.no_grad():
        outputs1 = bert_model(input_ids1)
        outputs2 = bert_model(input_ids2)
        embeddings1 = outputs1.last_hidden_state[:, 0, :]
        embeddings2 = outputs2.last_hidden_state[:, 0, :]

    similarity_score = cosine_similarity(embeddings1, embeddings2)
    return similarity_score

def read_and_run_specific_benchmark(model_name, csv_file, text_key, summary_key):
    print("Running benchmark: ", csv_file)
    total_benchmarks = 0
    sim_sum = 0
    csv.field_size_limit(10_000_000)
    reader = csv.DictReader(open(csv_file, encoding='utf-8'))
    for i, row in enumerate(reader):
        total_benchmarks = total_benchmarks + 1
        sim_sum = sim_sum + run_benchmark(model_name, row, text_key, summary_key)
    print(f"Average Similarity for {csv_file} Benchmark:", sim_sum/total_benchmarks)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarks with specified model.")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to use")
    args = parser.parse_args()

    model_name = args.model
    read_and_run_specific_benchmark(model_name, "data/benchmark_files/usb.csv", 'input_lines', 'output_lines')
    read_and_run_specific_benchmark(model_name, "data/benchmark_files/scraped_articles.csv", "text", "summary")