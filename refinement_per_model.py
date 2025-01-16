import os
import csv
from src.model_query import ModelQuery
from tqdm import tqdm

BASE_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "src/prompt_refinement.txt")
EXAMPLES_LTX = os.path.join(os.path.dirname(__file__), "src/prompts_LTX-Video.txt")

cathegory = "people"
CSV_FILE = os.path.join(os.path.dirname(__file__), f"captions_after_pipeline/captions_{cathegory}.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), f"output/LTX/captions_{cathegory}_refined.csv")

def main():

    with open(BASE_PROMPT_PATH, 'r') as base_file:
        base_prompt = base_file.read()

    with open(EXAMPLES_LTX, 'r') as examples_file:
        examples_prompt = examples_file.read()
    full_prompt = base_prompt + "\n" + examples_prompt

    input_captions = {}
    with open(CSV_FILE, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        input_captions = {row["filename"]: row["description"] for row in csv_reader}

    processed_files = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as output_file:
            csv_reader = csv.DictReader(output_file)
            processed_files = {row["filename"] for row in csv_reader}

    caption_data = []

    with open(OUTPUT_FILE, "a") as output_file:
        csv_writer = csv.DictWriter(output_file, fieldnames=["filename", "description"])
        if not processed_files:  # Write header only if the file is empty
            csv_writer.writeheader()

    # Adding tqdm progress bar
    for video, caption in tqdm(input_captions.items(), desc="Processing Captions", unit="caption"):
        if video in processed_files:
            print(video, " is already been processed, skipping...." )
            continue
        model_query = ModelQuery()
        refined_caption = model_query.query_model(
            system_prompt=full_prompt,
            input_file=str(caption)
        )

        # Remove escape characters from the refined caption
        sanitized_caption = refined_caption.replace("\\", "")  # Remove backslashes
        sanitized_caption = sanitized_caption.replace("\n", " ").replace("\r", " ").replace("\t", " ")

        caption_data.append({"filename": video, "caption": sanitized_caption})
        with open(OUTPUT_FILE, "a") as output_file:
            csv_writer = csv.DictWriter(output_file, fieldnames=["filename", "description"])
            csv_writer.writerow({"filename": video, "description": sanitized_caption})

if __name__ == "__main__":
    main()
