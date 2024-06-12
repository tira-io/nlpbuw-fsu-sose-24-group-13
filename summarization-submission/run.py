import os
from pathlib import Path
import pandas as pd
from transformers import pipeline
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

def generate_summary(model, text, max_length=1024):
    inputs = model.tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    output_ids = model.model.generate(**inputs, max_length=130, min_length=30, do_sample=False)
    summary = model.tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return summary

if __name__ == "__main__":
    # Load the data
    tira = Client()
    df = tira.pd.inputs("nlpbuw-fsu-sose-24", "summarization-validation-20240530-training").set_index("id")

    # Initialize the summarization pipeline
    summarizer = pipeline(task="summarization", model="facebook/bart-large-cnn")

    # Generate summaries
    df["summary"] = df["story"].apply(lambda text: generate_summary(summarizer, text))
    df = df.drop(columns=["story"]).reset_index()

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    output_path = Path(output_directory) / "predictions.jsonl"
    df.to_json(output_path, orient="records", lines=True)