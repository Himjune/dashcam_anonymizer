import os
import glob
import json
import subprocess
import multiprocessing

# FIX: required when frozen with PyInstaller on Windows — without it every
# worker process would re-run the whole script (endless restart loop).
multiprocessing.freeze_support()

import cv2
import pybboxes as pbx
import yaml
import argparse
from ultralytics import YOLO
import shutil
from rich.console import Console
from rich.progress import track
from natsort import natsorted
from os.path import join as osj

parser = argparse.ArgumentParser()
parser.add_argument("--config", help = "path of the training configuartion file", required = True)
args = parser.parse_args()
console = Console()


console.print(f"Reading the Configuration file from {args.config}", style="bold green")
with open(args.config, 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)


console.print("Loading YOLO Model...", style="bold green")
model = YOLO(config["model_path"])

if(config["generate_detections"]):
    # FIX: remove stale detection folders, otherwise ultralytics writes to
    # runs/detect/yolo_videos_pred2, pred3, ... and the script below finds nothing
    if os.path.exists("runs/"):
        shutil.rmtree("runs/")
    if os.path.exists("annot_jsons/"):
        shutil.rmtree("annot_jsons/")
    console.print("Generating YOLO Detections for the Videos", style="bold green")
    # Note: The YOLO model() call is generally capable of finding all supported video formats in a directory.
    # No changes are needed here.
    if(config["gpu_avail"]):
        console.print("GPU Available, Running on GPU", style="bold green")
        _ = model(source=config['videos_path'],
                save=False,
                save_txt=True,
                conf=config['detection_conf_thresh'],
                device='cuda:0',
                # FIX: absolute path — in ultralytics >=8.1 a relative project is
                # resolved against the global SETTINGS['runs_dir'], not the cwd
                project=os.path.abspath('runs/detect'),
                name="yolo_videos_pred")
    else:
        console.print("GPU Not Available, Running on CPU")
        _ = model(source=config['videos_path'],
                save=False,
                save_txt=True,
                conf=config['detection_conf_thresh'],
                device='cpu',
                # FIX: absolute path — in ultralytics >=8.1 a relative project is
                # resolved against the global SETTINGS['runs_dir'], not the cwd
                project=os.path.abspath('runs/detect'),
                name="yolo_videos_pred")
    
# =========================================================================================
# CHANGE 1: Search for multiple video file extensions, not just .mp4
# =========================================================================================
video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv'] # Add any other video formats you use
videos = []
for ext in video_extensions:
    videos.extend(glob.glob(os.path.join(config['videos_path'], ext)))

videos = natsorted(videos)
# =========================================================================================

if(config["generate_jsons"]):
    print(f"Generating JSONs for {len(videos)} videos")
    for video in track(videos):
        # =========================================================================================
        # CHANGE 2: Use os.path.splitext to robustly get the video name without its extension
        # =========================================================================================
        vid_name, _ = os.path.splitext(os.path.basename(video))
        # =========================================================================================

        vid = cv2.VideoCapture(video)
        height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        vid.release() # Release the video capture object after getting dimensions
        
        data_dict = {}
        # FIX: use the newest yolo_videos_pred* folder (ultralytics may append 2, 3, ...)
        pred_dirs = natsorted(glob.glob('runs/detect/yolo_videos_pred*'))
        labels_dir = osj(pred_dirs[-1], 'labels') if pred_dirs else ''
        annot_dir = natsorted(glob.glob(osj(labels_dir, f'{vid_name}_*.txt'))) if labels_dir else []

        try:
            for file in annot_dir:
                if (os.path.basename(file).endswith('.txt')):
                    # FIX: take the LAST underscore-separated part as the frame number.
                    # split("_")[1] broke on video names containing "_" (e.g. IMG_5808_42.txt
                    # returned 5808 for every frame, collapsing all labels into one frame).
                    frame_num = int(os.path.basename(file).replace(".txt","").rsplit("_", 1)[1])
                    with open(file, 'r') as fin:
                        for line in fin.readlines():
                            line = [float(item) for item in line.split()[1:]]
                            line = pbx.convert_bbox(line, from_type="yolo", to_type="voc", image_size=(width,height))
                            if(frame_num not in data_dict.keys()):
                                data_dict[frame_num] = [] # Initialize as empty list
                            data_dict[frame_num].append(line)
            if(not os.path.exists("annot_jsons/")):
                os.mkdir("annot_jsons")
            with open("annot_jsons/"+str(vid_name)+".json", 'w') as f:
                json.dump(data_dict, f)
            if not data_dict:
                console.print(f"WARNING: no detections found for {vid_name} — output will NOT be blurred!", style="bold red")
            else:
                console.print(f"{vid_name}: detections on {len(data_dict)} frames", style="green")
        except Exception as e:
            print(f'Could not process annotations for {video}. Error: {e}')


def blur_regions(image, regions):
    """
    Blurs the image, given the x1,y1,x2,y2 cordinates using Gaussian Blur.
    """
    for region in regions:
        x1,y1,x2,y2 = region
        # FIX: expand the box by 15% so the blur covers the object edges
        mx, my = 0.15 * (x2 - x1), 0.15 * (y2 - y1)
        x1, y1, x2, y2 = x1 - mx, y1 - my, x2 + mx, y2 + my
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
        # Ensure coordinates are within image bounds
        y1, y2 = max(0, y1), min(image.shape[0], y2)
        x1, x2 = max(0, x1), min(image.shape[1], x2)
        if x1 < x2 and y1 < y2:
            roi = image[y1:y2, x1:x2]
            # Kernel size must be odd
            blur_k = config["blur_radius"] if config["blur_radius"] % 2 != 0 else config["blur_radius"] + 1
            blurred_roi = cv2.GaussianBlur(roi, (blur_k, blur_k), 0)
            image[y1:y2, x1:x2] = blurred_roi
    return image


if not(os.path.exists(config["output_folder"])):
    console.print(f"Creating Directory {config['output_folder']} to store the anonymized videos", style="bold green")
    os.mkdir(config["output_folder"])

anonymized_videos_path = config["output_folder"]

for video in track(videos):
    # =========================================================================================
    # CHANGE 3: Use os.path.splitext again for consistency
    # =========================================================================================
    vid_name, _ = os.path.splitext(os.path.basename(video))
    # =========================================================================================
    
    json_path = f'annot_jsons/{vid_name}.json'
    if(os.path.exists(json_path)):
        with open(json_path) as F:
            data = json.load(F)

            # FIX: temporal padding — the detector occasionally skips single frames,
            # so apply every detection to +-temporal_pad_frames neighbouring frames too
            pad = int(config.get("temporal_pad_frames", 5))
            if pad > 0:
                padded = {}
                for k, regions in data.items():
                    fnum = int(k)
                    for nf in range(fnum - pad, fnum + pad + 1):
                        padded.setdefault(str(nf), []).extend(regions)
                data = padded

            video_capture = cv2.VideoCapture(video)
            
            # =========================================================================================
            # CHANGE 4: The output video will always be an MP4 for compatibility with the 'avc1' codec.
            # =========================================================================================
            out_vid_path = osj(anonymized_videos_path, vid_name + '.mp4')
            # =========================================================================================
            
            frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_size = (frame_width, frame_height)
            
            fps = round(video_capture.get(cv2.CAP_PROP_FPS))
            # 'avc1' is a good choice for H.264 codec in an .mp4 container.
            output_video = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*'avc1'), fps, frame_size)
            if not output_video.isOpened():
                # FIX: fall back to mp4v if the avc1 encoder is unavailable in this OpenCV build
                output_video = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
            count = 1
            while True:
                ret, frame = video_capture.read()
                
                if not ret:
                    break
                
                if str(count) in data:
                    frame = blur_regions(frame, data[str(count)])

                output_video.write(frame)
                count+=1
            video_capture.release()
            output_video.release()
            # FIX: copy the audio track from the original (OpenCV drops it)
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                tmp_path = out_vid_path + ".tmp.mp4"
                r = subprocess.run([ffmpeg, "-y", "-loglevel", "error",
                                    "-i", out_vid_path, "-i", video,
                                    "-map", "0:v", "-map", "1:a?",
                                    "-c:v", "copy", "-c:a", "copy", "-shortest", tmp_path])
                if r.returncode == 0:
                    os.replace(tmp_path, out_vid_path)
                elif os.path.exists(tmp_path):
                    os.remove(tmp_path)
            else:
                console.print("ffmpeg not found — output saved without audio", style="yellow")
        print(f"Processed Video {vid_name}")
    else:
        console.print(f"No objects detected in file {video}, copying file as is.", style="bold orange")
        shutil.copy(video, anonymized_videos_path)
        console.print(f"Copied Video {vid_name}", style="bold green")
        
#remove runs folder
if os.path.exists("runs/"):
    console.print(f"Removing Temporary Files...")
    shutil.rmtree("runs/")
if os.path.exists("annot_jsons/"):
    shutil.rmtree("annot_jsons/")

console.print(f"Blurred Videos are stored in {anonymized_videos_path}", style="bold yellow")
