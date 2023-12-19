import argparse
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmpose.apis import multi_gpu_test, single_gpu_test
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet
from tqdm import tqdm
from time import time
import os.path as osp
import os

from models import *
from mmcv_custom import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", "-l", type=int, default=-1)
    parser.add_argument("--score", "-s", action="store_true")
    parser.add_argument(
            '--cfg-options',
            nargs='+',
            action=DictAction,
            default={},
            help='override some settings in the used config, the key-value pair '
            'in xxx=yyy format will be merged into config file. For example, '
            "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    eval_config = cfg.get('evaluation', {})
    batch_size = cfg.data.get('samples_per_gpu', 1)
    workers = cfg.data.get('workers_per_gpu', 1)
    if args.batch_size > 0:
        batch_size = args.batch_size
    if args.workers > 0:
        workers = args.workers
    if cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
        os.makedirs(cfg.work_dir, exist_ok=True)
    cfg.data.val.data_cfg.use_gt_bbox=True
    dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=batch_size,
        workers_per_gpu=workers,
        dist=False,
        shuffle=False)
    
    model = build_posenet(cfg.model)

    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    # model = fuse_conv_bn(model)
    model = model.cuda()
    model.eval()
    
    
    results = dict()
    
    if args.score:
        outputs = single_gpu_test(MMDataParallel(model, device_ids=[0]), data_loader)
        results.update(dataset.evaluate(outputs, cfg.work_dir, **eval_config))
        
    refined = 0
    start_tick = time()
    print(f"{torch.cuda.memory_allocated()/1024/1024:.2f}M")
    
    with torch.no_grad():
        bar = tqdm(data_loader)
        for i, data in enumerate(bar):
            if i == args.limit:
                break
            torch.cuda.synchronize()
            x = data['img'].cuda()
            output = model.backbone(x)

            if hasattr(model.backbone, 'quality_predictor'):
                _, (_, count) = output
                refined += count

            if model.with_neck:
                output = model.neck(output)
                if hasattr(model.neck, 'quality_predictor'):
                    _, (_, count) = output
                    refined += count

            output = model.keypoint_head(output)
            bar.set_description(f"{torch.cuda.memory_allocated()/1024/1024:.2f}M")
            torch.cuda.synchronize()
    total = time() - start_tick
    count = (len(data_loader)) * batch_size
    if args.limit != -1:
        count = batch_size * (args.limit)
    print(f'Refined: {refined} / {count} / {count - refined}')
    print(f'Throughput: {count/total:.2f}')
    
    results.update(dict(dropped=count-refined, throughput=count/total))
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()