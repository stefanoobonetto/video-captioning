import os
import cv2
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from tqdm import tqdm
import csv
from src.model_query import ModelQuery

PROMPT_PATH = os.path.join(os.getcwd(), "src/prompt.txt")
MAX_FRAME = 20

def load_frames_from_directory(directory_path):
    """
    Load frames (images) from a directory, selecting only the first 20 frames.
    """
    frames = []
    for file_name in sorted(os.listdir(directory_path))[:MAX_FRAME]:
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            frame_path = os.path.join(directory_path, file_name)
            frame = cv2.imread(frame_path)
            if frame is not None:
                frames.append(frame)
    return frames

def generate_caption(model, processor, frame):
    """
    Generate a caption for a single frame using BLIP.
    """
    inputs = processor(images=frame, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    caption = model.generate(**inputs)
    return processor.decode(caption[0], skip_special_tokens=True)

def folder_captioning(folder_path, model, processor):
    """
    Generate captions for a folder of frames.
    """
    frames = load_frames_from_directory(folder_path)
    captions = []
    print("Generating captions for frames...")
    for frame in tqdm(frames):
        caption = generate_caption(model, processor, frame)
        captions.append(caption)
    return captions

def load_existing_captions(csv_file):
    """
    Load existing captions from the CSV file.
    Returns a set of filenames already processed.
    """
    existing_files = set()
    if os.path.exists(csv_file):
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                existing_files.add(row['filename'])
    return existing_files

if __name__ == "__main__":
    print("Loading models...")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    redo_hunyuan = {
        "landscapes_50" : ["130_072wpvM7aS8", "365_tNcqs0Rr4e8", "386_4R9HpESkor8", "423_0ncnJ3hkVVE", "601_HaCqJzOlfkE", "499_CyiMlcRKE9w", "717_roaVZPLWqtM", "998_SSdHUdoaeJg", "1124_-fib1FfnpkQ", "1888_7FUWFMSlGfM", "2260_Gh2yGSTIiSI", "1899_84IR-sL2iI0", "1047_CjUyIs7-IXQ", "1273_ytl1MQC_9rE", "1899_84IR-sL2iI0"], 
        "animals_50" : ["305_6A59ikvOi00", "572_jW0aFgDYM4c", "1053_MLWQ-qvra68", "1151_4d-ItHAojEU", "1212_x8IQeCtHSvE", "1560__FEG2Q256yg", "2364_wJzN3rfubH8"],
        "people_50" : ["406_okPeQz2sxIc", "444_TJ32UVNGmtQ", "858_jQZ_PhvRE14", "1069_UK-q7sxB8oQ", "1128_b-ECRS4R1Po", "1170_9emjJlPsaFQ", "1253_TPxVj7do42I", "1317_EV4ZeuFVeMA", "1465_OEcIUErHwMA"],
        "vehicles_50" : ["852_4RLJyV8wj9o", "1006_Sz1is20e6IU", "1040_TRqxTqL3pL0", "1046_nxrMuKhLCns", "1206_Vu5adaBn314", "1328_fR3s12FZ9kI", "1347_ulKH1C7JoNw", "1635_JEj8EZAuq-o", "1721_BWr1xvKifuo", "2068_basNjtv1FH0", "2082_CNbld-7QJ7s", "2300_hVKgY1ilx0Y"],
        "objects_50" : ["118_tQw4cJh7LVw", "227_MU4DhM5lKe0", "329_2vubTJvigGA", "360_5Y1GpL768Sk", "431_Y2lV7w65GyE", "481_6l6nUnvDjz0", "627_c3RJEw5UnXE", "690_9wSNsTuXbfE", "752_Zy_4Oi8ie4I", "753_EQc38hHoCx4", "769_e5EMPg5Smb0", "792_rPvcsyBeekQ", "798_6a8pJBoMK2Q", "961_h-4Bse5FaeY", "980_ZIngSGeILZ4", "984_cfR1PVshyIc", "995_oUz8CP0AJiU", "1255_megGoHWei58", "1272_kDLzAZhFEVY", "1300_1ltOymZ8IeQ", "1514_THWHMyOQmwI", "2201_0mJaV398KCs", "2360_umdlxHU96J4", "2365_wK5QpAvdEgw"]
    }
    
    redo_cogvideo = {
        "vehicles_50": ["80_OmFN5BCbWeA", "174_hspS7jtuGFA", "660_-bUmb5LY_lc", "1206_Vu5adaBn314", "1721_BWr1xvKifuo", "2205_1eo2-7k_Wds"],
        "people_50": ["430_KZ0Zk1u-Yo8", "444_TJ32UVNGmtQ", "1069_UK-q7sxB8oQ", "2026_3amamm00pgI"],
        "animals_50": ["572_jW0aFgDYM4c", "884_1X6y255nQfA", "1151_4d-ItHAojEU", "2001_i3Yx_JHtXVU"],
        "objects_50": ["65_5GqERXir2rc", "66_927BvkIZglw", "329_2vubTJvigGA", "643_9u3XDSj_KVs", "698_c5kKOKfKNYE", "752_Zy_4Oi8ie4I", "980_ZIngSGeILZ4", "984_cfR1PVshyIc", "1255_megGoHWei58", "1300_1ltOymZ8IeQ", "2365_wK5QpAvdEgw"],
        "landscapes_50": ["130_072wpvM7aS8", "1047_CjUyIs7-IXQ", "1273_ytl1MQC_9rE", "1720_Af-YGwVlcXE", "1824_6G4vunxpKQs"]
    }

    model_query = ModelQuery()
    path_folders = os.path.join(os.path.dirname(__file__), "Dataset_50")
    os.makedirs(os.path.join(os.getcwd(), "output"), exist_ok=True)

    for folder in redo_cogvideo:
        video_folder = os.path.join(path_folders, folder)
        csv_file = os.path.join(os.getcwd(), f"output/video_captions_{folder}.csv")
        with open(csv_file, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=["filename", "caption"])
                
        # Load existing captions
        existing_files = load_existing_captions(csv_file)

        for video in redo_cogvideo[folder]:
            if video in existing_files:
                print(f"Skipping already processed video: {video}")
                continue

            video_path = os.path.join(video_folder, video, "origin")
            captions = folder_captioning(video_path, model=blip_model, processor=blip_processor)

            caption = model_query.query_model(
                system_prompt=PROMPT_PATH,
                input_file=str(captions)
            )
            captions_data = {"filename": video, "caption": caption}

            print(f"\n-----------------------------------Response from llama ({video})-----------------------------------\n")
            print(caption)
            print("\n-----------------------------------------------------------------------------------------\n")

            # Append the current video's caption to the CSV file
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=["filename", "caption"])
                writer.writerow(captions_data)


