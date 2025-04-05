# PaperClip README

## Finetuning
In order to run the finetuning script, run the following on a GPU:

`python3 backend/model/run_benchmark.py --mode finetune`

In order to run the benchmarks on the finetuned model from above, run the following. Do not forget to update the path to the newly finetuned model in the script:

`python3 backend/model/run_benchmark.py --mode benchmark`

## Running the Benchmarks
In order to run the benchmarks on any of the base models, use the following command:

`python3 backend/model/finetune.py --model [MODEL_NAME]`

## Web Application Specs

## Model Specs
The code related to the different models we used, their finetuning process and the benchmarks that they were tested against can all be found in the model/ directory. Here is a breakdown of the folder:

1. `pdf_scraper.py`: Scraper that traverses the [ArXiv dataset](https://www.kaggle.com/datasets/sumitm004/arxiv-scientific-research-papers-dataset) in data/arxiv_ai.csv, downloads each pdf, scrapes the text from the pdf and constructs a new dataset in `data/benchmark_files/filtered_articles.csv`
2. `run_benchmark.py`: Script that runs the two benchmarks in the `data/benchmark_files` directory and output the average similarity score
2. `/finetuning`
    - `gpu_finetuning.py`: File containing the code that was ran using a GPU to fine tune our model based off the [BillSum dataset](https://huggingface.co/datasets/FiscalNote/billsum?library=datasets)
    - `checkpoint-250/`: Folder contaiing the details of the finetuned LLama 3.2-1B model
3. `/data`:
    - `arxiv_ai.csv`: Raw CSV file of ArXiv AI pdf urls and their summaries
    - `/benchmark_files`:
        - `filtered_articles.csv`: CSV file for the benchmark containing around 2000 ArXiv AI research papers' text and respective summaries (scraped from `arxvi_ai.csv`)
        - `usb.csv`: CSV file for the [Unified Summarization Benchmark](https://huggingface.co/datasets/kundank/usb) which contains the text of wikipedia articles across 8 different domains and their respective summaries