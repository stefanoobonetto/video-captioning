# Video Captioning

This repository contains a Python script designed to generate textual captions for videos using an AI model.

I used it to generate captions starting from real videos to generate correspondent synthetic videos:

![Example Output](assets/pipeline.png)  

## How It Works

The `captioning.py` script processes videos located in the `Dataset_50` folder and generates a textual description (caption) for each video.  
It selects frames from each video (either the first `N` frames or `k` evenly spaced frames), generates a caption for each frame using a BLIP model, and then summarizes them into a single video description using a language model.  
The generated captions are saved in per-category CSV files under the `output` directory. Optionally, captions can be refined and stored in a separate folder.

## Instructions

1. **Install Dependencies**:  
   Ensure you have Python installed on your system. Install the required libraries by running:

   ```bash
   pip install -r requirements.txt
   ```
   
2. **Prepare Dataset**:  
   Place your videos (organized by category) inside the `Dataset_50` folder.  
   Each video must have its frames extracted under a subfolder named `origin`.

   Example folder structure:
   ```
   Dataset_50/
       vehicles_50/
           80_OmFN5BCbWeA/
               origin/
                   frame_0001.jpg
                   frame_0002.jpg
                   ...
       people_50/
           430_KZ0Zk1u-Yo8/
               origin/
                   frame_0001.jpg
                   frame_0002.jpg
                   ...
   ```

3. **Run the Script**:  
   Execute the script by running:

   ```bash
   python captioning.py --dataset-root Dataset_50 --output-root output --modality spaced --k 20
   ```

   You can choose:
   - `--modality spaced` (default) to use k evenly spaced frames  
   - `--modality consecutive` to use the first `max_frame` frames

   Example with refinement enabled:
   ```bash
   python captioning.py --refinement true
   ```

4. **View Results**:  
   The generated captions will be saved in per-category CSV files such as `output/video_captions_vehicles_50.csv`.  
   If refinement is enabled, refined CSVs will be stored under `output/LTX`.

---

### Additional Notes

a. **Model Used**:  
The script uses [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) (Bootstrapping Language-Image Pretraining) to generate frame-level captions.  
It then uses the `ModelQuery` class (from `src/model_query.py`) to query a language model and summarize frame captions into a single video description.

b. **Custom Prompt**:  
The refinement step uses two prompt files:
- `src/prompt_refinement.txt` – the base system prompt
- `src/prompts_LTX-Video.txt` – examples used to guide refinement

You can modify these files to customize the refinement behavior.

c. **CSV Format**:  
The resulting CSVs contain two columns:
- `filename`: the video ID
- `caption` or `description`: the generated description text

d. **Supported Image Formats**:  
Frame images must be in `.jpg`, `.jpeg`, or `.png` format.

e. **Support**  
For additional help or information, consult the documentation within the repository or contact the repository's maintainer.
