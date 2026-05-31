# PaperClip

## Project Overview
Full-stack AI research assistant built with React, Flask, Hugging Face Transformers, SentenceTransformers, and GPT-based RAG pipelines for document summarization and user friendly contextual question answering

## Finetuning
In order to run the finetuning script, run the following on a CPU:

`python3 backend/pipeline/finetuning/cpu_finetuning.py`

On a GPU:

`python3 backend/pipeline/benchmark.py --mode finetune`

## Running the Benchmarks
In order to run the benchmarks on any of the base models, use the following command:

1. On openai model - `python3 backend/pipeline/benchmark.py --model openai`
2. On finetuned model - `python3 backend/pipeline/benchmark.py --model local`

## Web Application Specs

## Model Specs
The code related to the different models we used, their finetuning process and the benchmarks that they were tested against can all be found in the model/ directory. Here is a breakdown of the folder:

1. `backend/pipeline/scraper.py`: Scraper that traverses the [ArXiv dataset](https://www.kaggle.com/datasets/sumitm004/arxiv-scientific-research-papers-dataset) in data/arxiv_ai.csv, downloads each pdf, scrapes the text from the pdf and constructs a new dataset in `backend/pipeline/data/filtered_articles.csv`
2. `benchmark.py`: Script that runs the benchmark in the `data/benchmark_files` directory and outputs the average similarity score
2. `backend/pipeline/finetuning`
    - `gpu_finetuning.py`: File containing the code that was ran using a GPU to fine tune our model based off the [BillSum dataset](https://huggingface.co/datasets/FiscalNote/billsum?library=datasets)
    - `cpu_finetuning.py`: File containing the code that was ran using a CPU to fine tune our model based off the [BillSum dataset](https://huggingface.co/datasets/FiscalNote/billsum?library=datasets)
    - `checkpoint-250/`: Folder containing the details of the finetuned LLama 3.2-1B model on the GPU
    - `cpu_output/`: Folder containing the details of the finetuned GPT 4o mini model on the CPU
3. `backend/pipeline/data`:
    - `arxiv_ai.csv`: Raw CSV file of ArXiv AI pdf urls and their summaries
    - `/benchmark_files`:
        - `filtered_articles.csv`: CSV file for the benchmark containing around 2000 ArXiv AI research papers' text and respective summaries (scraped from `arxvi_ai.csv`)
        - `usb.csv`: CSV file for the [Unified Summarization Benchmark](https://huggingface.co/datasets/kundank/usb) which contains the text of wikipedia articles across 8 different domains and their respective summaries


