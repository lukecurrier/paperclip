import os
import pandas as pd
import requests
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


csv_file = "data/arxiv_ai.csv"
df = pd.read_csv(csv_file)

if "pdf_url" not in df.columns:
    raise ValueError("Dataset must contain a 'pdf' column with PDF URLs.")

def download_pdf(pdf_url, save_path="temp.pdf"):
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            file.write(response.content)
        return save_path
    
def extract_text(pdf_url):
    pdf_path = download_pdf(pdf_url)
    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(pdf_path)
    text, _, _ = text_from_rendered(rendered)
    os.remove(pdf_path)
    return text

if __name__ == '__main__':
    results = []
    for i, row in df.iterrows():
        if i > 2:
            break
        pdf_url = row["pdf_url"]
        try:
            text = extract_text(pdf_url)
            if text:
                summary = row["summary"]
                results.append({"text": text, "summary": summary})
        except:
            pass
    output_df = pd.DataFrame(results)
    output_df.to_csv("data/benchmarks/scraped_articles.csv", index=False)
    print("Scraping complete. Data saved to scraped_articles.csv")
