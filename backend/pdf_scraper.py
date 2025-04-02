import os
import pandas as pd
import requests
import nltk
from nltk.tokenize import sent_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from dotenv import load_dotenv
import re
import warnings
from pdfminer.high_level import extract_text

warnings.filterwarnings("ignore", category=UserWarning, module="pdfminer")


# Load Kaggle dataset
csv_file = "data/arxiv_ai.csv"  # Replace with actual dataset filename
df = pd.read_csv(csv_file)

# Ensure 'pdf' column exists
if "pdf_url" not in df.columns:
    raise ValueError("Dataset must contain a 'pdf' column with PDF URLs.")

def download_pdf(pdf_url, save_path="temp.pdf"):
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            file.write(response.content)
        return save_path
    else:
        raise Exception(f"Failed to download PDF: {response.status_code}")

def extract_text_pdfminer(pdf_url):
    pdf_path = download_pdf(pdf_url)
    text = extract_text(pdf_path)
    os.remove(pdf_path)
    return text


# Process dataset
results = []
for i, row in df.iterrows():
    if i > 10:
        break
    pdf_url = row["pdf_url"]
    text = extract_text_pdfminer(pdf_url)
    if text:
        summary = row["summary"]
        results.append({"text": text, "summary": summary})

# Save results to CSV
output_df = pd.DataFrame(results)
output_df.to_csv("data/scraped_articles.csv", index=False)

print("Scraping complete. Data saved to scraped_articles.csv")
