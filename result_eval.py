from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,time
import numpy as np
import time
import torch

from lib.utils.opts import opts

from lib.models.stNet import get_det_net, load_model, save_model
from lib.dataset.coco_bhdata import COCO
from lib.dataset.coco_eval import COCOeval
from lib.external.nms import soft_nms

from lib.utils.decode import ctdet_decode
from lib.utils.post_process import ctdet_post_process

import cv2

from progress.bar import Bar
#TP 


CONFIDENCE_thres = 0.6
COLORS = [(255, 0, 0)]

FONT = cv2.FONT_HERSHEY_SIMPLEX

def cv2_demo(frame, detections):
    det = []
    for i in range(detections.shape[0]):
        if detections[i, 4] >= CONFIDENCE_thres:
            pt = detections[i, :]
            cv2.rectangle(frame,(int(pt[0])-4, int(pt[1])-4),(int(pt[2])+4, int(pt[3])+4),COLORS[0], 2)
            cv2.putText(frame, str(pt[4]), (int(pt[0]), int(pt[1])), FONT, 1, (0, 255, 0), 1)
            det.append([int(pt[0]), int(pt[1]),int(pt[2]), int(pt[3]),detections[i, 4]])
    return frame, det

def VideoWriter_init(show_flag):
    if not show_flag:
        return None
    videoName = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '.avi'
    videoName = os.path.join("/home/cg/fengrubei/DFSNET/Moving-object-detection/result_video", videoName)
    fps = 10
    size = (512, 512)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter(videoName, fourcc,fps,size) if show_flag else None
    return videoWriter
def Eval(dataset_coco,result_path,iou_type="bbox"):
    
    coco = dataset_coco
    coco_dets = coco.loadRes(f'{result_path}')
    print(f"coco_dets: {coco_dets}")
    coco_eval = COCOeval(coco, coco_dets,iou_type)

    # 感觉重点可以改这里 不用IoU值进行衡量
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

def test(split, result_path):    
    dataset = COCO(opt, split)
    Eval(dataset.coco,result_path ,iou_type="distance")
    Eval(dataset.coco,result_path ,iou_type="bbox")

if __name__ == '__main__':
    opt = opts().parse()
    
    split = opt.test_split

    show_flag = opt.show_results

    results_path = "/home/cg/fengrubei/DFSNET/Moving-object-detection/weights/simdata_original/DLAFormerTemporal/depth:1_batch_size_12/results/results_latest.json"
    test(split, results_path)