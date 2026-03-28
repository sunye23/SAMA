import json
import os
import cv2
import math
import multiprocessing as mp
from tqdm import tqdm

VIDEO_DIR = "/data/data-pool/sunye/dataset/Sa2VA-Training/vidstg_videos"
ANNOTATION_DIR = "/data/data-pool/sunye/dataset/Sa2VA-Training/vidstg_jsons"
RAW_FRAMES_ROOT = "/data/data-pool/sunye/dataset/Sa2VA-Training/vidstg_videos_frames"

FRAME_INTERVAL = 4
NUM_PROCESSES = 8

os.makedirs(RAW_FRAMES_ROOT, exist_ok=True)


def process_video_list(args):
    video_list, chunk_idx = args
    local_results = {}

    pbar = tqdm(video_list, desc=f"Worker {chunk_idx}", position=chunk_idx, leave=True)
    for video_filename in pbar:
        vid = os.path.splitext(video_filename)[0]
        annotation_file = os.path.join(ANNOTATION_DIR, f"{vid}.json")
        video_path = os.path.join(VIDEO_DIR, video_filename)

        if not os.path.exists(annotation_file):
            continue

        with open(annotation_file, "r") as f:
            ann_data = json.load(f)

        trajectories = ann_data.get("trajectories", [])
        subject_objects = ann_data.get("subject/objects", [])

        tid_category_map = {}
        for obj in subject_objects:
            tid = obj["tid"]
            category = obj.get("category", "unknown")
            tid_category_map[tid] = category

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        raw_save_dir = os.path.join(RAW_FRAMES_ROOT, vid)
        os.makedirs(raw_save_dir, exist_ok=True)

        new_annotation = {
            "video_id": ann_data.get("video_id", vid),
            "width": ann_data.get("width"),
            "height": ann_data.get("height"),
            "fps": ann_data.get("fps"),
            "frame_count": ann_data.get("frame_count"),
            "subject/objects": subject_objects,
            "frames": []
        }

        current_frame_idx = 0
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame_idx < len(trajectories):
                objs_this_frame = trajectories[current_frame_idx]
            else:
                objs_this_frame = []

            raw_filename = f"frame_{current_frame_idx}.jpg"
            raw_path = os.path.join(raw_save_dir, raw_filename)
            cv2.imwrite(raw_path, frame)

            boxes_info = []
            for obj in objs_this_frame:
                bbox = obj["bbox"]
                xmin, ymin = bbox["xmin"], bbox["ymin"]
                xmax, ymax = bbox["xmax"], bbox["ymax"]
                tid = obj["tid"]
                cat = tid_category_map.get(tid, "unknown")

                boxes_info.append({
                    "tid": tid,
                    "category": cat,
                    "bbox": [xmin, ymin, xmax, ymax]
                })

            new_annotation["frames"].append({
                "frame_id": current_frame_idx,
                "raw_filename": raw_filename,
                "boxes": boxes_info
            })

            current_frame_idx += FRAME_INTERVAL
            if current_frame_idx >= total_frames:
                break

        cap.release()
        local_results[vid] = new_annotation

    return local_results


def main():
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
    video_files.sort()

    chunk_size = math.ceil(len(video_files) / NUM_PROCESSES)
    sublists = [
        (video_files[i*chunk_size:(i+1)*chunk_size], i)
        for i in range(NUM_PROCESSES)
    ]

    with mp.Pool(NUM_PROCESSES) as pool:
        results_list = []
        for partial_result in tqdm(
            pool.imap(process_video_list, sublists),
            desc="Overall Progress",
            total=NUM_PROCESSES,
            unit="chunk"
        ):
            results_list.append(partial_result)


if __name__ == "__main__":
    main()
