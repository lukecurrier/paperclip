import argparse
import os

def summarize(document_path):
    if not os.path.isfile(document_path):
        print(f"'{document_path}' not found.")
        return False
    
    load_dotenv()
    api_key = os.getenv("HUGGINGFACE_TOKEN")

    model_name = "meta-llama/Llama-3.1-8B-Instruct" 
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=api_key)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=api_key)

    with open(document_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    prompt = f"""Please summarize the following markdown content:
    
{text}

Summary:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=300,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
        )
    
    summary = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return summary
        
    