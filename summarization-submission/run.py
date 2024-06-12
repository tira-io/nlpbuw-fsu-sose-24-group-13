import os
from pathlib import Path
import pandas as pd
from transformers import pipeline
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

def generate_summary(summarizer, text, max_new_tokens=130):
    if len(text) > 1024:
        text = text[:1024]
    summary = summarizer(text, max_new_tokens=max_new_tokens, min_length=30, do_sample=False)[0]['summary_text']
    return summary

if __name__ == "__main__":
    # Load the data
    tira = Client()
    df = tira.pd.inputs("nlpbuw-fsu-sose-24", "summarization-validation-20240530-training").set_index("id")

    # Initialize the summarization pipeline
    summarizer = pipeline(task="summarization", model="sshleifer/distilbart-cnn-12-6")

    # Generate summaries
    df["summary"] = df["story"].apply(lambda text: generate_summary(summarizer, text))
    df = df.drop(columns=["story"]).reset_index()

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    output_path = Path(output_directory) / "predictions.jsonl"
    df.to_json(output_path, orient="records", lines=True)