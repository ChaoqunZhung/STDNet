from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from lib.dataset.cocoeval import COCOeval
import numpy as np
import json
import os
import sep
import torch.utils.data as data
import torch
import cv2
import math

from lib.utils.image import flip, color_aug
from lib.utils.image import get_affine_transform, affine_transform
from lib.utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from lib.utils.image import draw_dense_reg
from lib.utils.opts import opts
from lib.utils.augmentations import Augmentation
from astropy.io import fits


class COCO(data.Dataset):
    opt = opts().parse()
    num_classes = 1
    default_resolution = [512, 512]
    dense_wh = False
    reg_offset = True
    mean = np.array([0.49965, 0.49965, 0.49965], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.08255, 0.08255, 0.08255], dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super().__init__()

        self.opt = opt
        self.split = split

        self.img_dir0 = self.opt.data_dir
        self.img_dir = self.opt.data_dir

        if opt.test_real_data:
            if split != "test":
                self.resolution = [512, 512]
                self.annot_path = os.path.join(self.img_dir0, split + ".json")
            else:
                self.resolution = [512, 512]
                self.annot_path = "/home/cg/fengrubei/data/annotations_0405xiugai/test.json"
        else:
            self.resolution = [512, 512]
            self.annot_path = os.path.join(self.img_dir0, split + ".json")

        self.down_ratio = opt.down_ratio
        self.max_objs = opt.K
        self.seqLen = opt.seqLen

        self.class_name = ["star"]
        self._valid_ids = [1]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}

        print("==> initializing coco 2017 {} data.".format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print("Loaded {} {} samples".format(split, self.num_samples))

        self.aug = Augmentation() if split == "train" else None

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score)),
                    }
                    if len(bbox) > 5:
                        detection["extreme_points"] = list(
                            map(self._to_float, bbox[5:13])
                        )
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir, time_str):
        json.dump(
            self.convert_eval_format(results),
            open(f"{save_dir}/results_{time_str}.json", "w"),
        )
        print(f"{save_dir}/results_{time_str}.json")

    def run_eval(self, results, save_dir, time_str, iou_type="distance", save_result=True):
        if save_result:
            self.save_results(results, save_dir, time_str)

        coco_dets = self.coco.loadRes(f"{save_dir}/results_{time_str}.json")
        coco_eval = COCOeval(self.coco, coco_dets, iou_type)

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval.analyze_snr_recall(
            snr_bins=[
                ("Low", 0.0, 3.0),
                ("Medium", 3.0, 6.0),
                ("High", 6.0, 100.0),
            ],
            dist_thrs=[3, 5],
        )

        stats = coco_eval.stats
        precisions = coco_eval.eval["precision"]

        return stats, precisions

    def run_eval_per_snr_bins(
        self,
        results,
        save_dir,
        time_str,
        snr_bins=None,
        distance_thresholds=None,
        save_result=True,
    ):
        import copy as _copy

        if snr_bins is None:
            snr_bins = [
                ("Low", 0.0, 3.0),
                ("Medium", 3.0, 6.0),
                ("High", 6.0, 40.0),
            ]
        if distance_thresholds is None:
            distance_thresholds = [3, 5, 10]

        if save_result:
            self.save_results(results, save_dir, time_str)

        results_path = f"{save_dir}/results_{time_str}.json"
        coco_dets = self.coco.loadRes(results_path)

        try:
            with open(self.annot_path, "r") as f:
                gt_json = json.load(f)
        except Exception as e:
            print("Failed to load original annotation json:", e)
            return {}

        summary = {}

        for name, mn, mx in snr_bins:
            anns = []
            for ann in gt_json.get("annotations", []):
                snr = ann.get("snr", None)
                if snr is None:
                    continue

                ok = True
                if mn is not None and not (snr >= mn):
                    ok = False
                if mx is not None and not (snr < mx):
                    ok = False

                if ok:
                    anns.append(_copy.deepcopy(ann))

            if len(anns) == 0:
                summary[name] = {"gt_count": 0, "bins": {}}
                continue

            img_ids = {a["image_id"] for a in anns}
            imgs = [im for im in gt_json.get("images", []) if im["id"] in img_ids]

            gt_subset = {
                "images": imgs,
                "annotations": anns,
                "categories": gt_json.get("categories", []),
            }

            tmp_gt_path = os.path.join(
                save_dir, f'gt_{name.replace(" ", "_")}_{time_str}.json'
            )
            with open(tmp_gt_path, "w") as f:
                json.dump(gt_subset, f)

            cocoGt_sub = coco.COCO(tmp_gt_path)
            coco_eval = COCOeval(cocoGt_sub, coco_dets, "distance")

            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            summary[name] = {
                "gt_count": len(anns),
                "coco_stats": coco_eval.stats.tolist()
                if hasattr(coco_eval, "stats")
                else None,
            }

        out_path = os.path.join(save_dir, f"snr_eval_{time_str}.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Wrote per-SNR evaluation summary to {out_path}")
        return summary

    def run_eval_just(self, save_dir, time_str, iouth):
        coco_dets = self.coco.loadRes(f"{save_dir}/{time_str}")
        coco_eval = COCOeval(self.coco, coco_dets, "bbox", iouth=iouth)

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval.stats, coco_eval.eval["precision"]

    def _coco_box_to_bbox(self, box):
        return np.array(
            [box[0], box[1], box[0] + box[2], box[1] + box[3]],
            dtype=np.float32,
        )

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]["file_name"]

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        seq_num = self.seqLen
        imIdex = int(file_name.split(".")[0].split("/")[-1])

        base_path = "/".join(file_name.split("/")[:-1]) + "/"
        ext = "." + file_name.split(".")[-1]

        img = np.zeros([self.resolution[0], self.resolution[1], 3, seq_num])

        for ii in range(seq_num):
            imIndexNew = "%05d" % max(imIdex - ii, 1)
            imName = (base_path + imIndexNew + ext).replace(
                "home/cg/fengrubei", "/dev1/frb/DSFNet"
            )

            im_16bit = cv2.imread(imName, cv2.IMREAD_UNCHANGED)

            if im_16bit is not None and im_16bit.dtype == np.uint16:
                fits_img = im_16bit.astype(np.float32)

                bkg = sep.Background(fits_img)
                fits_img = fits_img - bkg
                fits_img[fits_img < 0] = 0

                fits_img = (fits_img - np.mean(fits_img)) / np.std(fits_img)
                fits_img = (
                    (fits_img - fits_img.min())
                    / (fits_img.max() - fits_img.min())
                    * 255
                ).astype(np.uint8)

                im = np.stack([fits_img] * 3, axis=-1)
            else:
                im = cv2.imread(imName)

            if ii == 0:
                imgOri = im

            inp_i = (im.astype(np.float32) / 255.0 - self.mean) / self.std
            img[:, :, :, ii] = inp_i

        bbox_tol, cls_id_tol = [], []

        for k in range(num_objs):
            ann = anns[k]
            bbox_tol.append(self._coco_box_to_bbox(ann["bbox"]))
            cls_id_tol.append(self.cat_ids[ann["category_id"]])

        inp = img.transpose(2, 3, 0, 1).astype(np.float32)

        height, width = img.shape[0], img.shape[1]
        c = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        s = max(height, width) * 1.0

        output_h = height // self.down_ratio
        output_w = width // self.down_ratio

        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        for k in range(num_objs):
            bbox = bbox_tol[k]
            cls_id = cls_id_tol[k]

            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            h = np.clip(h, 0, output_h - 1)
            w = np.clip(w, 0, output_w - 1)

            if h > 0 and w > 0:
                radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)))))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                    dtype=np.float32,
                )
                ct = np.clip(ct, [0, 0], [output_w - 1, output_h - 1])
                ct_int = ct.astype(np.int32)

                draw_umich_gaussian(hm[cls_id], ct_int, radius)

                wh[k] = (w, h)
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1

        ret = {
            "input": inp,
            "hm": hm,
            "reg_mask": reg_mask,
            "ind": ind,
            "wh": wh,
            "imgOri": imgOri,
            "file_name": file_name,
        }

        if self.reg_offset:
            ret["reg"] = reg

        return img_id, ret