from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lib.utils.logger import Logger
import time
import torch,os
from progress.bar import Bar
from lib.utils.data_parallel import DataParallel
from lib.utils.utils import AverageMeter
from lib.utils.decode import ctdet_decode
from lib.utils.post_process import ctdet_post_process
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
import lib.utils.misc as utils
import numpy as np
import io
from contextlib import redirect_stdout
from lib.external.nms import soft_nms
from lib.dataset.coco_eval import CocoEvaluator,get_coco_api_from_dataset,COCOeval

import torch.distributed as dist
def post_process(output, meta, num_classes=1, scale=1):
    # decode
    hm = output['hm'].sigmoid_()
    wh = output['wh']
    reg = output['reg']

    torch.cuda.synchronize()
    dets = ctdet_decode(hm, wh, reg=reg)
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        dets[0][j][:, :4] /= scale
    return dets[0]

def merge_outputs(detections, num_classes ,max_per_image):
    results = {}
    for j in range(1, num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

        soft_nms(results[j], Nt=0.5, method=2)

    scores = np.hstack(
      [results[j][:, 4] for j in range(1, num_classes + 1)])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        # print(batch['input'].shape)
        outputs = self.model(batch['input'])
        loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, loss_stats
    

class BaseTrainer(object):
    def __init__(
            self, opt, model,optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModelWithLoss(model, self.loss)
        self.logger = Logger(opt)
        self.local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
        

    def set_device(self, gpus, device):
        if len(gpus) > 1:
            # self.model_with_loss = DataParallel(
            #     self.model_with_loss, device_ids=gpus).to(device)
            self.model_with_loss = self.model_with_loss.cuda()
            self.model_with_loss = SyncBatchNorm.convert_sync_batchnorm(self.model_with_loss)
            self.model_with_loss = DistributedDataParallel(self.model_with_loss, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()
            
        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader)
        # num_iters = 1
        # bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, (im_id, batch) in enumerate(data_loader):
            if iter_id >= num_iters:
              break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta' and k != 'file_name':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)
            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            out_str = 'phase=%s, epoch=%5d, iters=%d/%d,time=%0.4f, loss=%0.4f, hm_loss=%0.4f, wh_loss=%0.4f, off_loss=%0.4f' \
                  % (phase, epoch,iter_id+1,num_iters, time.time() - end,
                     loss.mean().cpu().detach().numpy(),
                     loss_stats['hm_loss'].mean().cpu().detach().numpy(),
                     loss_stats['wh_loss'].mean().cpu().detach().numpy(),
                     loss_stats['off_loss'].mean().cpu().detach().numpy())
            if self.local_rank ==0:

                print(out_str)
                self.logger.write(out_str+"\n")
            end = time.time()

            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
            del output, loss, loss_stats

        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = 1 / 60.

        return ret, results
    
    
    def run_multi_gpu_eval(self, phase, epoch, data_loader, dataset, device=None):
        model_with_loss = self.model_with_loss
        opt = self.opt # Get opt from self

        world_size = 1
        local_rank = 0 # Default for non-distributed
        is_distributed = len(opt.gpus) > 1 and dist.is_initialized()

        if is_distributed:
            model_with_loss = self.model_with_loss.module
            world_size = dist.get_world_size()
            local_rank = self.local_rank # Assuming self.local_rank is set during __init__
        
        model_with_loss.eval()
        torch.cuda.empty_cache()

        # Ensure baseds is the COCO ground truth API object
       
        coco_evaluator = COCOeval(iouType='distance')
        baseds = dataset.coco
        all_results_dict_rank = {} 
        coco_predictions_rank = []

        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader)
        end = time.time()

        for iter_id, (im_id_tensor, batch) in enumerate(data_loader):
            if iter_id >= num_iters: # Should not be needed if data_loader length is correct
              break
            data_time.update(time.time() - end)
            
            current_im_ids_np = im_id_tensor.numpy().astype(np.int32)

            for k_batch in batch:
                if k_batch != 'meta' and k_batch != 'file_name': # Check if 'meta' could be a list of dicts
                    batch[k_batch] = batch[k_batch].to(device=opt.device, non_blocking=True)
            
            outputs_batch, loss, loss_stats_batch = model_with_loss(batch)
            loss = loss.mean() # Ensure loss is reduced if necessary for logging per iter

            batch_size = batch['input'].size(0)
            for i in range(batch_size):
                current_im_id = current_im_ids_np[i]
                
                single_output_dict = {}
                for key, value_tensor in outputs_batch.items():
                    single_output_dict[key] = value_tensor[i:i+1] 

                # Construct meta for post_process for a single image from the batch
                # This part needs to be accurate based on how your dataset provides meta information
                # Assuming batch['meta'] is a list of meta dicts, or can be adapted
                # The original code constructed meta based on input tensor dims, which might be okay if consistent.
                inp_height_b, inp_width_b = batch['input'].shape[3], batch['input'].shape[4]
                c_val_b = np.array([inp_width_b / 2., inp_height_b / 2.], dtype=np.float32)
                s_val_b = max(inp_height_b, inp_width_b) * 1.0
                
                # opt.num_classes and opt.down_ratio should be available in self.opt
                meta_for_post = {
                    'c': c_val_b, 's': s_val_b,
                    'out_height': inp_height_b // getattr(opt, 'down_ratio', 4),
                    'out_width': inp_width_b // getattr(opt, 'down_ratio', 4)
                }
                # If batch['meta'] contains per-image 'c' and 's' or other necessary fields, use them.
                # E.g., if batch['meta'] is a list: current_meta_item = batch['meta'][i]
                # Then use current_meta_item to build meta_for_post

                dets_single_img = post_process(single_output_dict, meta_for_post, 
                                               num_classes=getattr(opt, 'num_classes', 1), 
                                               scale=getattr(opt, 'scale', 1))
                
                ret_single_img = merge_outputs([dets_single_img], 
                                               num_classes=getattr(opt, 'num_classes', 1), 
                                               max_per_image=opt.K)
                
                all_results_dict_rank[current_im_id] = ret_single_img

                # for class_id_val, detections_in_class_val in ret_single_img.items():
                #     # TODO: Map class_id_val to COCO category_id if necessary
                #     # coco_category_id = dataset.class_to_coco_cat[class_id_val] (example)
                #     coco_category_id = class_id_val # Assuming direct mapping for now
                #     for det_idx_val in range(detections_in_class_val.shape[0]):
                #         bbox_val = detections_in_class_val[det_idx_val, :4]
                #         score_val = detections_in_class_val[det_idx_val, 4]
                #         coco_predictions_rank.append({
                #             'image_id': int(current_im_id),
                #             'category_id': int(coco_category_id), 
                #             'bbox': [float(bbox_val[0]), float(bbox_val[1]), 
                #                      float(bbox_val[2] - bbox_val[0]), float(bbox_val[3] - bbox_val[1])], # x,y,w,h
                #             'score': float(score_val)
                #         })
            
            batch_time.update(time.time() - end)
            if local_rank == 0:
                print_str = 'phase=%s, epoch=%5d, iters=%d/%d,time=%0.4f, loss=%0.4f' % \
                      (phase, epoch, iter_id + 1, num_iters, batch_time.val, loss.item())
                for l_name in self.loss_stats:
                    if l_name in loss_stats_batch:
                         print_str += f', {l_name}={loss_stats_batch[l_name].mean().item():0.4f}'
                print(print_str)
            end = time.time()

            for l_stat_name in avg_loss_stats:
                if l_stat_name in loss_stats_batch:
                    avg_loss_stats[l_stat_name].update(
                        loss_stats_batch[l_stat_name].mean().item(), batch['input'].size(0))
            del outputs_batch, loss, loss_stats_batch

        # --- Gather results from all processes ---
        #gathered_coco_predictions_list = [None] * world_size
        gathered_custom_results_list = [None] * world_size

        if is_distributed:
            dist.barrier() # Ensure all processes finish inference
            #dist.all_gather_object(gathered_coco_predictions_list, coco_predictions_rank)
            dist.all_gather_object(gathered_custom_results_list, all_results_dict_rank)
        else:
            #gathered_coco_predictions_list[0] = coco_predictions_rank
            gathered_custom_results_list[0] = all_results_dict_rank
        
        final_coco_predictions_for_eval = []
        final_custom_results_for_eval = {}

        if local_rank == 0:
            # for preds_list in gathered_coco_predictions_list:
            #     if preds_list:
            #         final_coco_predictions_for_eval.extend(preds_list)
            for res_dict in gathered_custom_results_list:
                if res_dict:
                    final_custom_results_for_eval.update(res_dict)

        # --- Perform evaluation on rank 0 ---
        eval_summary_str = ""
        stats1_output = None
        ap50_metric = 0.0

        if local_rank == 0:
            # if final_coco_predictions_for_eval:
            #     try:
            #         cocoDt_obj = baseds.loadRes(final_coco_predictions_for_eval)
            #         coco_evaluator.cocoDt = cocoDt_obj
            #         # Ensure imgIds and catIds in params are updated if necessary
            #         # coco_evaluator.params.imgIds = sorted(cocoDt_obj.getImgIds()) 
            #         # coco_evaluator.params.catIds = sorted(cocoDt_obj.getCatIds()) # If using categories

            #         coco_evaluator.evaluate()
            #         coco_evaluator.accumulate()
                    
                    
            #         s_io = io.StringIO()
            #         with redirect_stdout(s_io):
            #             coco_evaluator.summarize()
            #         eval_summary_str = s_io.getvalue()
            #         print("COCO Evaluator Summary (Distance):")
            #         print(eval_summary_str)
            #     except Exception as e:
            #         print(f"Error during COCO evaluation: {e}")
            # else:
            #     print("No COCO predictions to evaluate.")

            if final_custom_results_for_eval:
                try:
                    stats1_output, _ = dataset.run_eval(final_custom_results_for_eval, opt.save_results_dir, 'latest')
                    if stats1_output is not None and len(stats1_output) > 1:
                         ap50_metric = stats1_output[1] # Assuming AP at IoU=0.50 is the second element
                    print(f"Custom run_eval stats: {stats1_output}")
                except Exception as e:
                    print(f"Error during dataset.run_eval: {e}")
            else:
                print("No custom results for dataset.run_eval.")

        # Populate `ret` dictionary
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = batch_time.avg # More meaningful time

        # Broadcast AP50 from rank 0 to other ranks
        if is_distributed:
            ap50_tensor = torch.tensor(ap50_metric, device=opt.device)
            dist.broadcast(ap50_tensor, src=0)
            if local_rank != 0:
                ap50_metric = ap50_tensor.item()
        
        ret['ap50'] = ap50_metric

        if local_rank == 0:
            return ret, final_custom_results_for_eval, stats1_output
        else:
            return ret, {}, None 
        
        
    def run_eval_epoch(self, phase, epoch, data_loader, dataset,device=None):
        model_with_loss = self.model_with_loss

        if len(self.opt.gpus) > 1:
            model_with_loss = self.model_with_loss.module
        model_with_loss.eval()
        torch.cuda.empty_cache()
   
        
        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader)
        end = time.time()

        for iter_id, (im_id, batch) in enumerate(data_loader):
            if iter_id >= num_iters:
              break
            data_time.update(time.time() - end)
            
            for k in batch:
                if k != 'meta' and k != 'file_name':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)
            output, loss, loss_stats = model_with_loss(batch)

            inp_height, inp_width = batch['input'].shape[3],batch['input'].shape[4]
            c = np.array([inp_width / 2., inp_height / 2.], dtype=np.float32)
            s = max(inp_height, inp_width) * 1.0

            meta = {'c': c, 's': s,
                    'out_height': inp_height,
                    'out_width': inp_width}

            dets = post_process(output, meta)
            ret = merge_outputs([dets], num_classes=1, max_per_image=opt.K)
            results[im_id.numpy().astype(np.int32)[0]] = ret

            loss = loss.mean()
            batch_time.update(time.time() - end)
            
            print('phase=%s, epoch=%5d, iters=%d/%d,time=%0.4f, loss=%0.4f, hm_loss=%0.4f, wh_loss=%0.4f, off_loss=%0.4f' \
                  % (phase, epoch,iter_id+1,num_iters, time.time() - end,
                     loss.mean().cpu().detach().numpy(),
                     loss_stats['hm_loss'].mean().cpu().detach().numpy(),
                     loss_stats['wh_loss'].mean().cpu().detach().numpy(),
                     loss_stats['off_loss'].mean().cpu().detach().numpy()))
            end = time.time()

            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
            del output, loss, loss_stats

        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        # coco_evaluator.accumulate()
        # coco_evaluator.summarize()
        stats1, _ = dataset.run_eval(results, opt.save_results_dir, 'latest')
        ret['time'] = 1 / 60.
        ret['ap50'] = stats1[1]

        return ret, results, stats1

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError

    def val(self, epoch, data_loader, dataset):
        # return self.run_epoch('val', epoch, data_loader)

        return self.run_eval_epoch('val', epoch, data_loader, dataset)
    def multi_gpu_val(self, epoch, data_loader, dataset, device=None):
        return self.run_multi_gpu_eval('val', epoch, data_loader, dataset, device)
    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)
    