import os
import cv2
import csv
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor
from src.model_query import ModelQuery


def ensure_parent_dir(path: str):
    """
    Ensure the parent directory for a file path exists.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)


def write_csv_header_if_missing(csv_path: str, fieldnames):
    """
    Create CSV and write header if file doesn't exist or is empty.
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        ensure_parent_dir(csv_path)
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def load_existing_filenames(csv_path: str, key: str = "filename"):
    """
    Return a set of 'filename' values from an existing CSV (used to skip processed items).
    """
    existing = set()
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, mode='r', newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and key in reader.fieldnames:
                for row in reader:
                    existing.add(row[key])
    return existing


def load_k_spaced_frames_from_directory(directory_path, k, exts=('.jpg', '.jpeg', '.png')):
    """
    Select exactly k frames uniformly distributed across N available frames (one approximately every N/k).
    """
    files = [f for f in sorted(os.listdir(directory_path)) if f.lower().endswith(exts)]
    N = len(files)
    if N == 0 or k <= 0:
        return []

    k = min(k, N)
    indices = np.linspace(0, N - 1, num=k)
    indices = np.round(indices).astype(int)

    frames = []
    for i in indices:
        path = os.path.join(directory_path, files[i])
        img_bgr = cv2.imread(path)
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            frames.append(img_rgb)
    return frames


def load_frames_from_directory(directory_path, max_frame, exts=('.jpg', '.jpeg', '.png')):
    """
    Load the first max_frame frames (alphabetical order).
    """
    frames = []
    for file_name in sorted(os.listdir(directory_path))[:max_frame]:
        if file_name.lower().endswith(exts):
            frame_path = os.path.join(directory_path, file_name)
            img_bgr = cv2.imread(frame_path)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                frames.append(img_rgb)
    return frames


def generate_caption(model, processor, frame, device):
    """
    Generate a caption for a single frame using BLIP.
    """
    with torch.no_grad():
        inputs = processor(images=frame, return_tensors="pt").to(device)
        caption_ids = model.generate(**inputs)  # optionally: max_new_tokens=30, num_beams=3
    return processor.decode(caption_ids[0], skip_special_tokens=True)


def folder_captioning(folder_path, model, processor, modality='consecutive', k=20, max_frame=20, device='cpu'):
    """
    Generate frame-level captions for a folder of frames.

    Parameters
    ----------
    modality : str
        'consecutive' -> use the first max_frame frames
        'spaced'      -> use k evenly spaced frames across the folder
    """
    if not os.path.isdir(folder_path):
        print(f"Missing folder: {folder_path} — skipping.")
        return []

    if modality == 'consecutive':
        frames = load_frames_from_directory(folder_path, max_frame=max_frame)
    elif modality == 'spaced':
        frames = load_k_spaced_frames_from_directory(folder_path, k=k)
    else:
        raise ValueError("modality must be 'consecutive' or 'spaced'")

    if not frames:
        print(f"No frames found in: {folder_path} — skipping.")
        return []

    captions = []
    print(f"Generating captions for frames... (mode: {modality}, count: {len(frames)})")
    for frame in tqdm(frames, leave=False):
        cap_txt = generate_caption(model, processor, frame, device)
        captions.append(cap_txt)
    return captions


def sanitize_text(text: str) -> str:
    """
    Remove escape characters and collapse whitespace for safer CSV output.
    """
    if text is None:
        return ""
    s = text.replace("\\", " ")
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = " ".join(s.split())
    return s


def refine_csv(
    input_csv: str,
    output_csv: str,
    base_prompt_path: str,
    examples_prompt_path: str,
    input_text_column_candidates=("caption", "description"),
):
    """
    Refine captions in input_csv using ModelQuery and write to output_csv with columns:
    filename, description

    - It accepts either 'caption' or 'description' as the source column for the input CSV.
    - Skips rows already present in output_csv.
    """

    with open(base_prompt_path, 'r', encoding='utf-8') as f:
        base_prompt = f.read()
    with open(examples_prompt_path, 'r', encoding='utf-8') as f:
        examples_prompt = f.read()
    full_prompt = base_prompt + "\n" + examples_prompt

    write_csv_header_if_missing(output_csv, fieldnames=["filename", "description"])
    processed = load_existing_filenames(output_csv, key="filename")

    if not os.path.exists(input_csv) or os.path.getsize(input_csv) == 0:
        print(f"[Refinement] Input CSV missing or empty: {input_csv}")
        return

    with open(input_csv, 'r', newline='', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames or "filename" not in reader.fieldnames:
            print(f"[Refinement] 'filename' column missing in {input_csv}")
            return

        text_col = None
        for cand in input_text_column_candidates:
            if cand in reader.fieldnames:
                text_col = cand
                break

        if text_col is None:
            print(f"[Refinement] No text column found in {input_csv}. Expected one of {input_text_column_candidates}.")
            return

        rows = list(reader)

    with open(output_csv, 'a', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["filename", "description"])
        for row in tqdm(rows, desc=f"Refining {os.path.basename(input_csv)}", unit="cap"):
            video = row["filename"]
            if video in processed:
                continue

            raw_caption = row[text_col]
            if not raw_caption:
                continue

            mq = ModelQuery()
            refined = mq.query_model(system_prompt=full_prompt, input_file=str(raw_caption))
            sanitized = sanitize_text(refined)
            writer.writerow({"filename": video, "description": sanitized})


def main():
    parser = argparse.ArgumentParser(description="Captioning + optional refinement pipeline")
    # Captioning options
    parser.add_argument("--dataset-root", type=str, default=os.path.join(os.path.dirname(__file__), "Dataset_50"),
                        help="Root folder containing category subfolders and per-video 'origin' frame folders")
    parser.add_argument("--output-root", type=str, default=os.path.join(os.getcwd(), "output"),
                        help="Where caption CSVs will be written")
    parser.add_argument("--modality", type=str, choices=["consecutive", "spaced"], default="spaced",
                        help="Frame selection modality")
    parser.add_argument("--k", type=int, default=20, help="Number of frames when modality='spaced'")
    parser.add_argument("--max-frame", type=int, default=20, help="Number of frames when modality='consecutive'")

    # Refinement options
    parser.add_argument("--refinement", type=lambda x: str(x).lower() in ("true", "1", "yes"),
                        default=False, help="If True, run refinement after captioning")
    parser.add_argument("--base-prompt", type=str, default=os.path.join(os.path.dirname(__file__), "src/prompt_refinement.txt"),
                        help="Path to base refinement prompt")
    parser.add_argument("--examples-prompt", type=str, default=os.path.join(os.path.dirname(__file__), "src/prompts_LTX-Video.txt"),
                        help="Path to examples prompt")
    parser.add_argument("--ltx-output-root", type=str, default=os.path.join(os.path.dirname(__file__), "output/LTX"),
                        help="Where refined CSVs will be written")

    args = parser.parse_args()

    # Example categories 
    redo_cogvideo = {
        "vehicles_50": ["80_OmFN5BCbWeA", "174_hspS7jtuGFA", "660_-bUmb5LY_lc", "1206_Vu5adaBn314", "1721_BWr1xvKifuo", "2205_1eo2-7k_Wds"],
        "people_50": ["430_KZ0Zk1u-Yo8", "444_TJ32UVNGmtQ", "1069_UK-q7sxB8oQ", "2026_3amamm00pgI"],
        "animals_50": ["572_jW0aFgDYM4c", "884_1X6y255nQfA", "1151_4d-ItHAojEU", "2001_i3Yx_JHtXVU"],
        "objects_50": ["65_5GqERXir2rc", "66_927BvkIZglw", "329_2vubTJvigGA", "643_9u3XDSj_KVs", "698_c5kKOKfKNYE", "752_Zy_4Oi8ie4I", "980_ZIngSGeILZ4", "984_cfR1PVshyIc", "1255_megGoHWei58", "1300_1ltOymZ8IeQ", "2365_wK5QpAvdEgw"],
        "landscapes_50": ["130_072wpvM7aS8", "1047_CjUyIs7-IXQ", "1273_ytl1MQC_9rE", "1720_Af-YGwVlcXE", "1824_6G4vunxpKQs"]
    }

    '''redo_hunyuan = {
        "landscapes_50" : ["130_072wpvM7aS8", "365_tNcqs0Rr4e8", "386_4R9HpESkor8", "423_0ncnJ3hkVVE", "601_HaCqJzOlfkE", "499_CyiMlcRKE9w", "717_roaVZPLWqtM", "998_SSdHUdoaeJg", "1124_-fib1FfnpkQ", "1888_7FUWFMSlGfM", "2260_Gh2yGSTIiSI", "1899_84IR-sL2iI0", "1047_CjUyIs7-IXQ", "1273_ytl1MQC_9rE", "1899_84IR-sL2iI0"], 
        "animals_50" : ["305_6A59ikvOi00", "572_jW0aFgDYM4c", "1053_MLWQ-qvra68", "1151_4d-ItHAojEU", "1212_x8IQeCtHSvE", "1560__FEG2Q256yg", "2364_wJzN3rfubH8"],
        "people_50" : ["406_okPeQz2sxIc", "444_TJ32UVNGmtQ", "858_jQZ_PhvRE14", "1069_UK-q7sxB8oQ", "1128_b-ECRS4R1Po", "1170_9emjJlPsaFQ", "1253_TPxVj7do42I", "1317_EV4ZeuFVeMA", "1465_OEcIUErHwMA"],
        "vehicles_50" : ["852_4RLJyV8wj9o", "1006_Sz1is20e6IU", "1040_TRqxTqL3pL0", "1046_nxrMuKhLCns", "1206_Vu5adaBn314", "1328_fR3s12FZ9kI", "1347_ulKH1C7JoNw", "1635_JEj8EZAuq-o", "1721_BWr1xvKifuo", "2068_basNjtv1FH0", "2082_CNbld-7QJ7s", "2300_hVKgY1ilx0Y"],
        "objects_50" : ["118_tQw4cJh7LVw", "227_MU4DhM5lKe0", "329_2vubTJvigGA", "360_5Y1GpL768Sk", "431_Y2lV7w65GyE", "481_6l6nUnvDjz0", "627_c3RJEw5UnXE", "690_9wSNsTuXbfE", "752_Zy_4Oi8ie4I", "753_EQc38hHoCx4", "769_e5EMPg5Smb0", "792_rPvcsyBeekQ", "798_6a8pJBoMK2Q", "961_h-4Bse5FaeY", "980_ZIngSGeILZ4", "984_cfR1PVshyIc", "995_oUz8CP0AJiU", "1255_megGoHWei58", "1272_kDLzAZhFEVY", "1300_1ltOymZ8IeQ", "1514_THWHMyOQmwI", "2201_0mJaV398KCs", "2360_umdlxHU96J4", "2365_wK5QpAvdEgw"]
    }'''
    
    print("Loading BLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    blip_model.eval()

    os.makedirs(args.output_root, exist_ok=True)

    for category, video_ids in redo_cogvideo.items():
        caption_csv = os.path.join(args.output_root, f"video_captions_{category}.csv")
        write_csv_header_if_missing(caption_csv, fieldnames=["filename", "caption"])

        already = load_existing_filenames(caption_csv, key="filename")

        for vid in tqdm(video_ids, desc=f"Captioning {category}", unit="video"):
            if vid in already:
                continue

            frames_dir = os.path.join(args.dataset_root, category, vid, "origin")
            captions = folder_captioning(
                frames_dir,
                model=blip_model,
                processor=blip_processor,
                modality=args.modality,
                k=args.k,
                max_frame=args.max_frame,
                device=device
            )
            if not captions:
                continue

            mq = ModelQuery()
            raw_combined_caption = mq.query_model(
                system_prompt="Summarize the following frame-level captions into a single concise video description.",
                input_file=str(captions)
            )
            combined_caption = sanitize_text(raw_combined_caption)

            with open(caption_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["filename", "caption"])
                writer.writerow({"filename": vid, "caption": combined_caption})

        if args.refinement:
            refined_csv = os.path.join(args.ltx_output_root, f"captions_{category}_refined.csv")
            ensure_parent_dir(refined_csv)
            refine_csv(
                input_csv=caption_csv,
                output_csv=refined_csv,
                base_prompt_path=args.base_prompt,
                examples_prompt_path=args.examples_prompt,
                input_text_column_candidates=("caption", "description"),
            )


if __name__ == "__main__":
    main()

    
