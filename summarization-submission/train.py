import json
from transformers import pipeline

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

# Load the test set articles
with open("test_text.jsonl", "r") as f:
    test_articles = [json.loads(line) for line in f]

# Generate summaries
predictions = []
for article in test_articles:
    summary = summarizer(article['story'], max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    predictions.append({
        "id": article["id"],
        "summary": summary
    })

# Write predictions to file
with open("predictions.jsonl", "w") as f:
    for prediction in predictions:
        f.write(json.dumps(prediction) + "\n")
