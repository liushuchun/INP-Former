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

# repo-specific imports (must exist in your repo)
from utils import evaluation_batch, setup_seed, get_logger
from dataset import get_data_transforms
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
from torch.nn.init import trunc_normal_

warnings.filterwarnings("ignore")


# ---------- Dataset for meta.json ----------
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
        img_pil = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img_pil.size

        # transformed tensor for model
        if self.transform:
            img_t = self.transform(img_pil)
        else:
            img_t = TF.to_tensor(img_pil)

        # placeholder mask tensor (transformed) if GT mask exists
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

        # original image as numpy uint8 RGB for overlay/save
        img_orig_np = np.array(img_pil)  # H x W x 3 RGB uint8

        return img_t, mask_t, rel_path, cls_name, img_orig_np, (orig_w, orig_h)


# ---------- util functions ----------
def normalize_to_uint8(x):
    """
    Convert numpy or torch array to 0-255 uint8 (H x W).
    Accepts float arrays in [0,1] or [0,255] or any numeric array.
    """
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    arr = np.array(x, dtype=np.float32)
    arr = arr - arr.min()
    maxv = arr.max() if arr.size > 0 else 0.0
    if maxv > 0:
        arr = arr / maxv
    arr = (arr * 255.0).astype('uint8')
    return arr


def apply_colormap_and_overlay(orig_rgb_input, heat_uint8, alpha=0.5):
    """
    Robustly accept various types for orig_rgb_input and heat_uint8.
    Returns: heat_color_bgr (uint8 BGR), overlay_rgb (uint8 RGB)
    """
    # ---- handle orig image ----
    orig = orig_rgb_input
    # unwrap list/tuple
    if isinstance(orig, (list, tuple)):
        orig = orig[0]
    # if torch tensor
    if 'torch' in str(type(orig)):
        orig = orig.detach().cpu().numpy()
    # if numpy but channels-first
    if isinstance(orig, np.ndarray):
        if orig.ndim == 3 and orig.shape[0] == 3:
            orig = np.transpose(orig, (1, 2, 0))
        # if float image in [0,1]
        if orig.dtype != np.uint8:
            tmp = np.clip(orig, 0, 255)
            if tmp.max() <= 1.0:
                tmp = (tmp * 255.0).astype('uint8')
            else:
                tmp = tmp.astype('uint8')
            orig = tmp
    else:
        orig = np.array(orig, dtype='uint8')

    # ensure shape HxWx3
    if orig.ndim == 2:
        orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
    if orig.ndim == 3 and orig.shape[2] == 4:
        orig = orig[:, :, :3]

    # ---- handle heat map ----
    heat = heat_uint8
    if isinstance(heat, (list, tuple)):
        heat = heat[0]
    if 'torch' in str(type(heat)):
        heat = heat.detach().cpu().numpy()
    if isinstance(heat, np.ndarray) and heat.dtype != np.uint8:
        heat = normalize_to_uint8(heat)
    # ensure 2D uint8
    if heat.ndim == 3 and heat.shape[2] == 3:
        heat = cv2.cvtColor(heat, cv2.COLOR_RGB2GRAY)
    heat_uint8 = heat.astype('uint8')

    # resize heat map to orig if sizes mismatch
    if (heat_uint8.shape[0] != orig.shape[0]) or (heat_uint8.shape[1] != orig.shape[1]):
        heat_uint8 = cv2.resize(heat_uint8, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)

    # colormap & overlay
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)  # BGR
    orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(heat_color, float(alpha), orig_bgr, 1.0 - float(alpha), 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return heat_color, overlay_rgb


# ---------- main ----------
def main(args):
    setup_seed(1)

    # transforms
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)

    # locate meta.json
    meta_file_candidate_1 = os.path.join(args.data_path, 'meta.json')
    meta_file_candidate_2 = os.path.join(args.data_path, 'test', 'meta.json')
    if os.path.exists(meta_file_candidate_2):
        meta_file = meta_file_candidate_2
    elif os.path.exists(meta_file_candidate_1):
        meta_file = meta_file_candidate_1
    else:
        raise FileNotFoundError(f"Cannot find meta.json under {args.data_path} or {os.path.join(args.data_path,'test')}")

    # build dataset/dataloader
    test_dataset = RealIADMetaDataset(root=args.data_path, meta_file=meta_file, transform=data_transform,
                                     gt_transform=gt_transform, phase='test', default_mask_size=256)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # build model (same config as before)
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
                       target_layers=[2, 3, 4, 5, 6, 7, 8, 9], remove_class_token=True,
                       fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]], fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
                       prototype_token=INP)
    model = model.to(device)

    # load weights (look in save_dir/save_name first, fallback to ./model.pth)
    state_path = os.path.join(args.save_dir, args.save_name, 'model.pth')
    if not os.path.exists(state_path):
        if os.path.exists('model.pth'):
            state_path = 'model.pth'
        else:
            raise FileNotFoundError(f"model.pth not found at {state_path} or ./model.pth")
    model.load_state_dict(torch.load(state_path), strict=True)
    model.eval()

    # prepare outputs
    results_root = os.path.join(args.save_dir, args.save_name, 'results')
    os.makedirs(results_root, exist_ok=True)
    meta_out_list = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Real-IAD inference'):
            img_t, mask_t, rel_path, cls_name, img_orig_np, orig_size = batch

            # unwrap singletons robustly
            if isinstance(rel_path, (list, tuple)): rel_path = rel_path[0]
            if isinstance(cls_name, (list, tuple)): cls_name = cls_name[0]
            if isinstance(img_orig_np, (list, tuple)): img_orig_np = img_orig_np[0]

            # ensure numpy uint8 HxWx3
            if 'torch' in str(type(img_orig_np)):
                img_orig_np = img_orig_np.detach().cpu().numpy()
            if isinstance(img_orig_np, np.ndarray) and img_orig_np.ndim == 3 and img_orig_np.shape[0] == 3:
                img_orig_np = np.transpose(img_orig_np, (1, 2, 0))
            if isinstance(img_orig_np, np.ndarray) and img_orig_np.dtype != np.uint8:
                tmp = np.clip(img_orig_np, 0, 255)
                if tmp.max() <= 1.0:
                    tmp = (tmp * 255.0).astype('uint8')
                else:
                    tmp = tmp.astype('uint8')
                img_orig_np = tmp

            orig_h, orig_w = int(img_orig_np.shape[0]), int(img_orig_np.shape[1])

            # input tensor
            img_input = img_t.to(device)
            if img_input.ndim == 3:
                img_input = img_input.unsqueeze(0)

            # forward
            out = model(img_input)
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                en = out[0]
                de = out[1]
            else:
                en = out
                de = None

            # anomaly map generation (simple; replace with evaluation_batch internal logic if desired)
            if isinstance(en, (list, tuple)):
                feat = en[-1]
            else:
                feat = en

            if isinstance(feat, torch.Tensor):
                if feat.ndim == 4:
                    amap = torch.norm(feat, p=2, dim=1).cpu().numpy()  # [B, Hf, Wf]
                    amap = amap[0]
                elif feat.ndim == 3:
                    amap = feat.mean(dim=2).cpu().numpy()[0]
                else:
                    amap = feat.squeeze().cpu().numpy()
            else:
                raise RuntimeError("Unexpected feature type from model")

            # resize anomaly map back to original image size
            amap_resized = cv2.resize(amap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            amap_uint8 = normalize_to_uint8(amap_resized)  # HxW uint8

            # prepare output directories
            cls_dir = os.path.join(results_root, cls_name)
            os.makedirs(cls_dir, exist_ok=True)

            # build safe base name that includes parent folders flattened to avoid collisions
            base_filename = os.path.splitext(os.path.basename(rel_path))[0]
            rel_folder = os.path.dirname(rel_path).replace('\\', '/').strip('/')
            if rel_folder == '':
                safe_base = base_filename
            else:
                safe_base = rel_folder.replace('/', '__') + '__' + base_filename

            # save gray heatmap
            gray_name = f"{safe_base}_heat_gray.png"
            gray_path = os.path.join(cls_dir, gray_name)
            cv2.imwrite(gray_path, amap_uint8)

            # color heatmap and overlay
            heat_color_bgr, overlay_rgb = apply_colormap_and_overlay(img_orig_np, amap_uint8, alpha=0.5)
            color_name = f"{safe_base}_heat_color.png"
            color_path = os.path.join(cls_dir, color_name)
            cv2.imwrite(color_path, heat_color_bgr)  # BGR

            overlay_name = f"{safe_base}_overlay.png"
            overlay_path = os.path.join(cls_dir, overlay_name)
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(overlay_path, overlay_bgr)

            # binarize using Otsu on the uint8 anomaly map
            try:
                _, bin_mask = cv2.threshold(amap_uint8, 0, 255, cv2.THRESH_OTSU)
            except Exception:
                _, bin_mask = cv2.threshold(amap_uint8, 0, 255, cv2.THRESH_BINARY)

            bin_mask_uint8 = bin_mask.astype('uint8')

            # determine anomaly: any non-zero pixel
            detected_anomaly = 1 if np.any(bin_mask_uint8 > 0) else 0

            # save mask in original image size only if anomaly detected
            if detected_anomaly:
                mask_name = f"{safe_base}_mask.png"
                mask_path_abs = os.path.join(cls_dir, mask_name)
                cv2.imwrite(mask_path_abs, bin_mask_uint8)
                rel_mask_path = os.path.relpath(mask_path_abs, results_root).replace('\\', '/')
            else:
                rel_mask_path = None

            # collect specie_name if present in original meta
            matched_meta = None
            for it in test_dataset.items:
                if it['rel_path'] == rel_path and it['cls_name'] == cls_name:
                    matched_meta = it['meta']
                    break
            specie_name = ""
            if matched_meta and 'specie_name' in matched_meta:
                specie_name = matched_meta.get('specie_name', "")

            # assemble meta entry
            meta_entry = {
                "img_path": rel_path,
                "mask_path": rel_mask_path,
                "cls_name": cls_name,
                "specie_name": specie_name,
                "anomaly": int(detected_anomaly)
            }
            meta_out_list.append(meta_entry)

    # write meta.json
    meta_out_file = os.path.join(results_root, 'meta.json')
    with open(meta_out_file, 'w', encoding='utf-8') as f:
        json.dump(meta_out_list, f, ensure_ascii=False, indent=2)

    print(f"Done. Meta written to: {meta_out_file}")
    print(f"Visuals and masks saved under: {results_root}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset root containing meta.json')
    parser.add_argument('--dataset', type=str, default='Real-IAD')
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
