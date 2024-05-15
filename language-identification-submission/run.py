from pathlib import Path

from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import preprocessing

if __name__ == "__main__":

    # Load the data
    tira = Client()
    
    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )
    text_validation["text"] = preprocessing.clean_text(text_validation["text"])


    df = text_validation.join(targets_validation.set_index("id"), on="id")

    # Load the model and make predictions
    model = load(Path(__file__).parent / "model.joblib")
    predictions = model.predict(df["text"])
    df["lang"] = predictions
    # df = df[["text", "lang"]]

    # Save the predictions
    # converting the prediction to the required format

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))

    # print(df.head(25))
    df[["id", "lang"]].to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )

