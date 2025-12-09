#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import argparse
import warnings
from functools import partial

import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

# repo-specific imports (assume same as before)
from utils import evaluation_batch, setup_seed, get_logger
from dataset import get_data_transforms
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
from torch.nn.init import trunc_normal_

warnings.filterwarnings("ignore")

# ---------- RealIADMetaDataset (same as before) ----------
class RealIADMetaDataset(torch.utils.data.Dataset):
    def __init__(self, root, meta_file, transform=None, gt_transform=None, phase="test", default_mask_size=256):
        super().__init__()
        self.root = root
        self.transform = transform
        self.gt_transform = gt_transform
        self.default_mask_size = default_mask_size

        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        assert phase in ['train', 'test']
        meta_section = meta.get(phase, meta.get("test", {}))

        self.items = []
        for cls_name, entries in meta_section.items():
            for e in entries:
                img_rel = e.get('img_path', '')
                mask_rel = e.get('mask_path', '') or ''
                img_path = os.path.join(root, img_rel)
                mask_path = os.path.join(root, mask_rel) if mask_rel else ''
                self.items.append({
                    'img_path': img_path,
                    'mask_path': mask_path,
                    'cls_name': e.get('cls_name', cls_name),
                    'meta': e,
                    'rel_path': img_rel
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        img_path = rec['img_path']
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img_t = self.transform(img)
        else:
            img_t = TF.to_tensor(img)

        # placeholder mask (0) or real mask if exists
        mask_t = torch.zeros(1, self.default_mask_size, self.default_mask_size)
        if rec['mask_path'] and os.path.exists(rec['mask_path']):
            mask = Image.open(rec['mask_path']).convert('L')
            if self.gt_transform:
                mask_t = self.gt_transform(mask)
            else:
                mask = mask.resize((self.default_mask_size, self.default_mask_size), Image.NEAREST)
                mask_t = TF.to_tensor(mask)
        rel_path = rec['rel_path']
        cls_name = rec['cls_name']
        return img_t, mask_t, rel_path, cls_name


# ---------- main (精简，只保留测试分支与模型构造核心) ----------
def main(args):
    setup_seed(1)
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)

    # find meta.json
    meta_file_candidate_1 = os.path.join(args.data_path, 'meta.json')
    meta_file_candidate_2 = os.path.join(args.data_path, 'test', 'meta.json')
    if os.path.exists(meta_file_candidate_2):
        meta_file = meta_file_candidate_2
    elif os.path.exists(meta_file_candidate_1):
        meta_file = meta_file_candidate_1
    else:
        raise FileNotFoundError(f"Cannot find meta.json under {args.data_path} or {os.path.join(args.data_path,'test')}")

    test_dataset = RealIADMetaDataset(root=args.data_path, meta_file=meta_file, transform=data_transform,
                                     gt_transform=gt_transform, phase='test', default_mask_size=256)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # model construction (same as before)
    encoder = vit_encoder.load(args.encoder)
    if 'small' in args.encoder:
        embed_dim, num_heads = 384, 6
    elif 'base' in args.encoder:
        embed_dim, num_heads = 768, 12
    elif 'large' in args.encoder:
        embed_dim, num_heads = 1024, 16
    else:
        raise ValueError("Architecture not in small/base/large")

    Bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)])
    INP = nn.ParameterList([nn.Parameter(torch.randn(args.INP_num, embed_dim)) for _ in range(1)])
    INP_Extractor = nn.ModuleList([Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                                    qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))])
    INP_Guided_Decoder = nn.ModuleList([Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8)) for _ in range(8)])
    model = INP_Former(encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor, decoder=INP_Guided_Decoder,
                       target_layers=[2,3,4,5,6,7,8,9], remove_class_token=True,
                       fuse_layer_encoder=[[0,1,2,3],[4,5,6,7]], fuse_layer_decoder=[[0,1,2,3],[4,5,6,7]],
                       prototype_token=INP)
    model = model.to(device)

    # load weights
    state_path =  'model.pth'
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"model.pth not found: {state_path}")
    model.load_state_dict(torch.load(state_path), strict=True)
    model.eval()

    # results root
    results_root = os.path.join(args.save_dir, args.save_name, 'results')
    os.makedirs(results_root, exist_ok=True)
    meta_out_list = []  # 将写入 meta.json 的条目集合

    def normalize_to_uint8(x):
        x = x - x.min()
        if x.max() > 0:
            x = x / x.max()
        x = (x * 255.).astype('uint8')
        return x

    with torch.no_grad():
        for img_t, mask_t, rel_path, cls_name in tqdm(test_dataloader, desc='Real-IAD inference'):
            # unwrap (dataloader returns batches)
            if isinstance(rel_path, (list, tuple)):
                rel_path = rel_path[0]
            if isinstance(cls_name, (list, tuple)):
                cls_name = cls_name[0]

            img_t = img_t.to(device)
            # forward - adapt to your model; expect (en, de, g_loss) or similar
            out = model(img_t)
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                en = out[0]
                de = out[1]
            else:
                en = out
                de = None

            # 简单示例：用最后一层 feature 的 channel-norm 作为 anomaly map（请根据你的 evaluation_batch 替换）
            if isinstance(en, (list, tuple)):
                feat = en[-1]
            else:
                feat = en
            if isinstance(feat, torch.Tensor):
                if feat.ndim == 4:
                    amap = torch.norm(feat, p=2, dim=1).cpu().numpy()  # [B, H, W]
                    amap = amap[0]
                elif feat.ndim == 3:
                    amap = feat.mean(dim=2).cpu().numpy()[0]  # simplistic
                else:
                    amap = feat.squeeze().cpu().numpy()
            else:
                raise RuntimeError("Unexpected feature type")

            # resize anomaly map -> 256x256
            amap_resized = cv2.resize(amap, (256, 256), interpolation=cv2.INTER_LINEAR)
            amap_uint8 = normalize_to_uint8(amap_resized)
            # 二值化（Otsu）
            try:
                _, bin_mask = cv2.threshold(amap_uint8, 0, 255, cv2.THRESH_OTSU)
            except Exception:
                _, bin_mask = cv2.threshold(amap_uint8, 0, 255, cv2.THRESH_BINARY)

            # 判断是否为异常：二值掩码上有非零像素 -> anomaly=1
            is_anomaly = 1 if np.any(bin_mask > 0) else 0

            # 保存 heatmap & mask（按类归档）
            cls_dir = os.path.join(results_root, cls_name)
            os.makedirs(cls_dir, exist_ok=True)
            # 生成文件名（把相对路径中的斜杠换为下划线，以避免目录嵌套）
            safe_base = rel_path.split("/")[-1]
            heatmap_name = f"{safe_base}_anom.png"
            heatmap_path = os.path.join(cls_dir, heatmap_name)
            cv2.imwrite(heatmap_path, amap_uint8)

            # 若异常则写 mask 文件并在 meta 中返回路径（相对 data_path）
            if is_anomaly:
                mask_name = f"{safe_base}_mask.png"
                mask_path_abs = os.path.join(cls_dir, mask_name)
                # save mask as binary PNG (0/255)
                cv2.imwrite(mask_path_abs, bin_mask)
                # produce relative path consistent with your sample, e.g. 'battery/NG/..._mask.png'
                # 我们把结果放在 save_dir/.../results/<cls>/<file>
                # 如果你需要把 mask 放回 data_path 下并用相对 data_path 的路径，请改写这里。
                # 使用相对到 args.save_dir/args.save_name/results 的路径：
                rel_mask_path = os.path.relpath(mask_path_abs, args.data_path).replace('\\','/')
            else:
                rel_mask_path = None

            # 全局 img score（用 anomaly map 的均值）
            img_score = float(amap_resized.mean())

            # 组装一条 meta 记录（和你示例一致）
            meta_entry = {
                "img_path": rel_path,              # 保持原始 meta 中的相对路径
                "mask_path": rel_mask_path,        # None -> JSON 会显示为 null
                "cls_name": cls_name,
                "specie_name": "",                 # 如果你的 meta 中有 specie_name，可以从原 meta 中取出；这里占位为空
                "anomaly": int(is_anomaly)
            }
            # 如果原 meta 中包含 specie_name（test_dataset.items），我们优先填入
            # 找到对应项的 original meta entry
            # 这里做一次尝试性填充
            matched_meta = None
            for it in test_dataset.items:
                if it['rel_path'] == rel_path and it['cls_name'] == cls_name:
                    matched_meta = it['meta']
                    break
            if matched_meta:
                if 'specie_name' in matched_meta:
                    meta_entry['specie_name'] = matched_meta.get('specie_name', "")
                # 如果原来有 mask_path 字段且为非空，也可以保持，但我们覆盖为推断结果的 mask_path
            meta_out_list.append(meta_entry)

    # 写入 meta.json（扁平数组）
    meta_out_file = os.path.join(results_root, 'meta.json')
    # 如果你想按 class 分组写入，可以改变结构。这里写成示例中那种一维数组（多条字典）
    with open(meta_out_file, 'w', encoding='utf-8') as f:
        json.dump(meta_out_list, f, ensure_ascii=False, indent=2)

    print(f"Done. Meta written to: {meta_out_file}")
    print(f"Anomaly masks (if any) and heatmaps saved under: {results_root}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Real-IAD')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='INP-Former-Multi-Class')
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--INP_num', type=int, default=6)
    parser.add_argument('--phase', type=str, default='test')
    args = parser.parse_args()

    args.save_name = args.save_name + f'_dataset={args.dataset}_Encoder={args.encoder}_Resize={args.input_size}_Crop={args.crop_size}_INP_num={args.INP_num}'
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    main(args)
