import re

def clean_text(input):
    output = []
    for text in input:
        text = text.lower()
        # Remove punctuation and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d', '', text)
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        output.append(text)
    return output
