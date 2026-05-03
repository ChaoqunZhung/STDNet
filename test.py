from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,time
import numpy as np
import time
import torch

from lib.utils.opts import opts

from lib.models.stNet import get_det_net, load_model, save_model
# from lib.dataset.coco_senet import COCO
from lib.dataset.coco_bhdata import COCO

from lib.external.nms import soft_nms

from lib.utils.decode import ctdet_decode
from lib.utils.post_process import ctdet_post_process

import cv2

from progress.bar import Bar
#TP 


CONFIDENCE_thres = 0.4
COLORS = [(240, 96, 0)]

FONT = cv2.FONT_HERSHEY_SIMPLEX

def draw_boxes(image, boxes, color=(50, 255, 0), label=None):
    img = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        # cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # 可以绘制label
        # if label:
        #     cv2.putText(img, label, (x1, y1 - 5), FONT, 0.6, color, 1)
        # 计算中心点
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # 半径取最大值
        radius = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2)

        # 绘制圆
        cv2.circle(img, (cx, cy), radius, color, thickness=2)
    return img


def cv2_demo(frame, detections):
    det = []
    for i in range(detections.shape[0]):
        if detections[i, 4] >= CONFIDENCE_thres:
            pt = detections[i, :]
            # cv2.rectangle(frame,(int(pt[0])-4, int(pt[1])-4),(int(pt[2])+4, int(pt[3])+4),COLORS[0], 2)
            # cv2.putText(frame, str(pt[4]), (int(pt[0]), int(pt[1])), FONT, 1, (0, 255, 0), 1)
            x1, y1 = int(pt[0]) - 4, int(pt[1]) - 4
            x2, y2 = int(pt[2]) + 4, int(pt[3]) + 4
            # 计算中心点
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            # 计算半径（对角线长度一半）
            radius = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2)
            # 画圆
            cv2.circle(frame, (cx, cy), radius, COLORS[0], 2)

            det.append([int(pt[0]), int(pt[1]),int(pt[2]), int(pt[3]),detections[i, 4]])
    return frame, det

def process(model, image, return_time):
    with torch.no_grad():
        output = model(image)[-1]
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output['reg']
        torch.cuda.synchronize()
        forward_time = time.time()
        dets = ctdet_decode(hm, wh, reg=reg)
    if return_time:
        return output, dets, forward_time
    else:
        return output, dets
    
def post_process(dets, meta, num_classes=1, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        dets[0][j][:, :4] /= scale
    return dets[0]

def pre_process(image, scale=1):
    height, width = image.shape[2:4]
    new_height = int(height * scale)
    new_width = int(width * scale)

    inp_height, inp_width = height, width
    c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0

    meta = {'c': c, 's': s,
            'out_height': inp_height ,
            'out_width': inp_width}
    return meta

def merge_outputs(detections, num_classes ,max_per_image):
    results = {}
    for j in range(1, num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

        soft_nms(results[j], Nt=0.3, method=2)

    scores = np.hstack(
      [results[j][:, 4] for j in range(1, num_classes + 1)])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results

def test(opt, split, modelPath, show_flag, results_name):

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    opt.save_results_dir = opt.save_results_dir.replace("results","imgs")

    # 保存推理图像的路径
    save_pred_dir = os.path.join(opt.save_results_dir, 'pred_vis')
    save_gt_dir = os.path.join(opt.save_results_dir, 'gt_vis')
    os.makedirs(save_pred_dir, exist_ok=True)
    os.makedirs(save_gt_dir, exist_ok=True)

    # Logger(opt)
    print(opt.model_name)

    dataset = COCO(opt, split)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    model = get_det_net({'hm': dataset.num_classes, 'wh': 2, 'reg': 2}, opt.model_name)  # 建立模型
    model = load_model(model, modelPath)
    model = model.cuda()
    
    model.eval()

    results = {}

    return_time = False
    scale = 1
    num_classes = dataset.num_classes
    max_per_image = opt.K

    # 将结果存储成视频
    # videoName = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '.avi'
    # videoName = os.path.join("/dev2/cg/fengrubei/DFSNET/Moving-object-detection/result_video", videoName)
    # fps = 10
    # size = (512, 512)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # videoWriter = cv2.VideoWriter(videoName, fourcc,fps,size) if show_flag else None

    num_iters = len(data_loader)
    bar = Bar('processing', max=num_iters)
    for ind, (img_id,pre_processed_images) in enumerate(data_loader):
        # print(ind)
        if(ind>len(data_loader)-1):
            break

        bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters,total=bar.elapsed_td, eta=bar.eta_td
        )

        # 读入图像
        detection = []
        meta = pre_process(pre_processed_images['input'], scale)
        image = pre_processed_images['input'].cuda()
        img = pre_processed_images['imgOri'].squeeze().numpy()

        # 检测
        output, dets = process(model, image, return_time)

        # 后处理
        dets = post_process(dets, meta, num_classes)
        detection.append(dets)
        ret = merge_outputs(detection, num_classes, max_per_image)

        if(show_flag):
            #保存
            frame, det = cv2_demo(img, ret[1])
            # videoWriter.write(frame)

            img_path = pre_processed_images['file_name'][0]
            # 假设 img_path 是字符串，例如 "/a/b/c/d/e/f.jpg"
            parts = img_path.split(os.sep)
            if opt.test_real_data:
                img_name = '_'.join(parts[-3:]).replace('.tif', '.jpg')
            else:
                img_name = '_'.join(parts[-3:])


            gt_boxes = pre_processed_images['gt_boxes']
            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes = gt_boxes.cpu().numpy()
            gt_boxes = gt_boxes.reshape(-1, 4)  # 保证 shape 是 (N, 4)


            # 绘制预测框图像
            frame_pred, det = cv2_demo(img, ret[1])  # ret[1] 为当前预测结果
            cv2.imwrite(os.path.join(save_pred_dir, img_name), frame_pred)

            # 绘制真实框图像
            frame_gt = draw_boxes(img, gt_boxes, color=(0, 0, 255), label='GT')
            cv2.imwrite(os.path.join(save_gt_dir, img_name), frame_gt)

            
            # frame, det = cv2_demo(img, dets[1])

            # cv2.imshow('frame',frame)
            # cv2.waitKey(5)

            # hm1 = output['hm'].squeeze(0).squeeze(0).cpu().detach().numpy()

            # cv2.imshow('hm', hm1)
            # cv2.waitKey(5)
        
        results[img_id.numpy().astype(np.int32)[0]] = ret
        
        bar.next()
    # videoWriter.release() if show_flag else None
    bar.finish()
    dataset.run_eval(results, opt.save_results_dir, results_name,"distance")
    dataset.run_eval(results, opt.save_results_dir, results_name,"bbox",save_result=False)

if __name__ == '__main__':
    opt = opts().parse()

    split = opt.test_split

    show_flag = opt.show_results

    if (not os.path.exists(opt.save_results_dir)):
        os.makedirs(opt.save_results_dir, exist_ok=True)

    if opt.load_model != '':
        modelPath = opt.load_model
    else:
        modelPath = './checkpoints/DSFNet.pth'

    print(modelPath)

    results_name = opt.model_name+'_'+modelPath.split('/')[-1].split('.')[0]
    
    test(opt, split, modelPath, show_flag, results_name)

##　命令：
'''
绘制图 + 视频：

python test.py --model_name Centerformer \
    --gpus 1 \
    --load_model /home/intern_5/zcq/DSFNet/Moving-object-detection/weights/sequence/10/Centerformer/2025_07_13_11_23_51/model_8.pth \
    --test_real_data true \
    --show_results True \
    --datasetname paint/sim \
    --data_dir /home/intern_5/zcq/DSFNet/data/annotations_buzhouzhen_hunhe \
    --seqLen 10

# 注意：/dev2/cg/fengrubei/DFSNET/Moving-object-detection/lib/dataset/coco_bhdata.py 第325和326行在训练的时候要注释掉 推理的时候加上去
'''