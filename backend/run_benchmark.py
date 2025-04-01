import os
from dotenv import load_dotenv
import torch
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import csv

load_dotenv()

bert_model = AutoModel.from_pretrained("bert-base-uncased")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
def run_benchmark(benchmark):
    #print("Running benchmark: ", benchmark)
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL")
    
    if hasattr(torch.backends, 'mps'):
        print(f"MPS available: {torch.backends.mps.is_available()}")
    
    client = OpenAI(api_key=api_key, base_url=base_url)

    SYSTEM_PROMPT = """You are a helpful assistant that summarizes scientific papers."""
    
    USER_PROMPT = f"""Please briefly summarize the following markdown content:
    
    {benchmark['text']}

    Summary:"""
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ],
        model="llama3p1-8b-instruct",
        temperature=0,
        max_tokens=300
    )    
    resp_content = response.choices[0].message.content
    tokenized_resp = bert_tokenizer.tokenize(resp_content)
    tokenized_benchmark = bert_tokenizer.tokenize(benchmark['summary'])
    
    #print(resp_content)
    #print(benchmark['summary'])
    
    input_ids1 = torch.tensor(bert_tokenizer.encode(resp_content, truncation=True, max_length=512)).unsqueeze(0)
    input_ids2 = torch.tensor(bert_tokenizer.encode(benchmark['summary'], truncation=True, max_length=512)).unsqueeze(0)

    # Obtain the BERT embeddings
    with torch.no_grad():
        outputs1 = bert_model(input_ids1)
        outputs2 = bert_model(input_ids2)
        embeddings1 = outputs1.last_hidden_state[:, 0, :]
        embeddings2 = outputs2.last_hidden_state[:, 0, :]

    similarity_score = cosine_similarity(embeddings1, embeddings2)
    print("Similarity Score:", similarity_score)
    return similarity_score


if __name__ == "__main__":
    total_benchmarks = 0
    sim_sum = 0
    reader = csv.DictReader(open('scisumm.csv', encoding='utf-8'))
    for i, row in enumerate(reader):
        if i >= 20:
            break
        total_benchmarks = total_benchmarks + 1
        sim_sum = sim_sum + run_benchmark(row)
    print("Average Similarity:", sim_sum/total_benchmarks)