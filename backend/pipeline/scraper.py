import os
import pandas as pd
import requests
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# CSV file with a dataset of ArXiv pdf urls and summaries
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

# Use marker pdf to extract text from the pdf
def extract_text(pdf_url):
    pdf_path = download_pdf(pdf_url)
    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(pdf_path)
    text, _, _ = text_from_rendered(rendered)
    os.remove(pdf_path)
    return text

# Construct a new CSV with the text from each of the pdfs and their corresponding summaries
if __name__ == '__main__':
    results = []
    for i, row in df.iterrows():
        pdf_url = row["pdf_url"]
        try:
            text = extract_text(pdf_url)
            if text:
                summary = row["summary"]
                results.append({"text": text, "summary": summary})
        except:
            pass
    output_df = pd.DataFrame(results)
    # Filter out the super long texts since the model cannot handle it
    filtered_df = output_df[output_df['text'].str.len() < 130000]
    filtered_df.to_csv("data/benchmark_files/filtered_articles.csv", index=False)
    print("Scraping complete. Data saved to scraped_articles.csv. Filtered data saced to filtered_articles.csv")
