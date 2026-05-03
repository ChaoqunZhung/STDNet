# -*- coding: utf-8 -*-
import os
import json
import random
from glob import glob
from PIL import Image

class YOLO2COCOTracker:
    def __init__(self, root_path, split_ratio=0.8, exts=("tif", "jpg", "png")):
        self.anno_count = 0
        self.image_id = 0
        self.root = root_path
        self.split_ratio = split_ratio
        self.exts = tuple(e.lower() for e in exts)
        self.coco_template = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "star"}],
            "videos": []
        }
        self._init_video_folders()
        self._split_videos()
        self._init_id_mappings()

    def _init_video_folders(self):
        self.video_folders = sorted(glob(os.path.join(self.root, "*")))
        self.video_info = {
            idx + 1: {"name": os.path.basename(v), "path": v}
            for idx, v in enumerate(self.video_folders)
        }

    def _split_videos(self):
        all_videos = list(self.video_info.keys())
        random.shuffle(all_videos)
        split_point = int(len(all_videos) * self.split_ratio)
        self.train_vids = all_videos[:split_point]
        self.val_vids = all_videos[split_point:]

    def _init_id_mappings(self):
        self.global_track_id = 1
        self.id_maps = {
            'train': {'image': {}, 'track': {}},
            'val': {'image': {}, 'track': {}}
        }

    def _list_images(self, video_path):
        files = []
        for ext in self.exts:
            files.extend(glob(os.path.join(video_path, "img", f"*.{ext}")))
            files.extend(glob(os.path.join(video_path, "img", f"*.{ext.upper()}")))
        return sorted(files)

    def _find_label(self, img_path):
        base = os.path.splitext(img_path)[0]
        label_path = base.replace(os.sep + "img" + os.sep, os.sep + "label" + os.sep) + ".txt"
        if os.path.exists(label_path):
            return label_path
        label_path = base.replace("/img/", "/label/") + ".txt"
        return label_path if os.path.exists(label_path) else None

    def _parse_labels(self, label_path, image_id, video_id, subset, img_w, img_h):
        annotations = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                snr_val = None
                if len(parts) == 5:
                    track_id = int(parts[0]) if parts[0].isdigit() else -1
                    x_center, y_center, w_rel, h_rel = map(float, parts[1:5])
                else:
                    try:
                        rels = list(map(float, parts[1:5]))
                        is_bbox = all(0.0 <= v <= 1.0 for v in rels)
                    except:
                        is_bbox = False

                    if is_bbox:
                        track_id = int(parts[0]) if parts[0].isdigit() else -1
                        x_center, y_center, w_rel, h_rel = rels
                        if len(parts) >= 6:
                            try:
                                snr_val = float(parts[5])
                            except:
                                snr_val = None
                    else:
                        track_id = -1
                        try:
                            x_center, y_center, w_rel, h_rel = map(float, parts[-4:])
                        except:
                            continue

                x_min = (x_center - w_rel / 2.0) * img_w
                y_min = (y_center - h_rel / 2.0) * img_h
                w = w_rel * img_w
                h = h_rel * img_h

                track_key = f"{video_id}_{track_id}" if track_id != -1 else None
                if track_key and track_key in self.id_maps[subset]['track']:
                    global_track_id = self.id_maps[subset]['track'][track_key]
                else:
                    global_track_id = self.global_track_id
                    if track_key:
                        self.id_maps[subset]['track'][track_key] = global_track_id
                    self.global_track_id += 1

                self.anno_count += 1

                ann = {
                    "id": self.anno_count,
                    "category_id": 1,
                    "image_id": image_id,
                    "track_id": -1,
                    "global_track_id": -1,
                    "bbox": [x_min, y_min, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "conf": 1.0,
                }
                if snr_val is not None:
                    ann["snr"] = float(snr_val)
                annotations.append(ann)

        return annotations

    def _process_video(self, video_id, subset):
        video_data = {"images": [], "annotations": []}
        video_path = self.video_info[video_id]["path"]

        img_files = self._list_images(video_path)
        video_len = len(img_files)

        video_entry = {
            "id": video_id,
            "file_name": self.video_info[video_id]["name"],
            "split": subset,
        }
        self.coco_template["videos"].append(video_entry)

        prev_id = -1
        for frame_idx, img_path in enumerate(img_files, 1):
            with Image.open(img_path) as img:
                width, height = img.size

            original_id = f"{video_id}_{frame_idx}"
            self.image_id += 1
            new_image_id = self.image_id
            self.id_maps[subset]['image'][original_id] = new_image_id

            image_entry = {
                "id": self.image_id,
                "file_name": img_path,
                "frame_id": frame_idx,
                "prev_image_id": prev_id,
                "next_image_id": new_image_id + 1 if frame_idx < video_len else -1,
                "video_id": video_id,
                "video_frame_id": frame_idx,
                "video_len": video_len,
            }
            video_data["images"].append(image_entry)
            prev_id = new_image_id

            label_path = self._find_label(img_path)
            if label_path and os.path.getsize(label_path) > 0:
                video_data["annotations"] += self._parse_labels(
                    label_path, new_image_id, video_id, subset, width, height
                )

        return video_data

    def convert(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        for subset, video_ids in [('train', self.train_vids), ('val', self.val_vids)]:
            self.anno_count = 0
            self.image_id = 0
            subset_data = self.coco_template.copy()
            subset_data["images"] = []
            subset_data["annotations"] = []

            for vid in video_ids:
                video_data = self._process_video(vid, subset)
                subset_data["images"].extend(video_data["images"])
                subset_data["annotations"].extend(video_data["annotations"])

            output_path = os.path.join(output_dir, f"{subset}.json")
            with open(output_path, 'w') as f:
                json.dump(subset_data, f, indent=2)

            print(f"{subset}: {len(subset_data['images'])} images, {len(subset_data['annotations'])} annotations")


if __name__ == "__main__":
    converter = YOLO2COCOTracker(
        root_path=r"/dev1/frb/DSFNet/data/0602_real_snr/",
        split_ratio=1,
        exts=("tif", "jpg", "png")
    )
    converter.convert(output_dir="data_splited")