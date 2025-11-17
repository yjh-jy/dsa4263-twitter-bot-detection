#!/usr/bin/env python3
"""
multimodal_model.py

Unified multimodal training script for Twitter bot detection.

This script:
- Loads multimodal data (text + profile image URL)
- Optionally performs async image prefetching into the cache
- Extracts image and text embeddings
- Trains multiple multimodal architectures:
    1) Text-only
    2) Image-only
    3) Concat Fusion
    4) GMU Fusion
    5) Cross-Modal Attention Fusion
- Evaluates on val/test splits
- Saves predictions, embeddings, and model weights

Typical usage:

1. Preprocess CSV dataset into train/val/test splits
   python preprocessing_multimodal.py \
       --input data/raw/twitter_human_bots_dataset.csv \
       --output-dir data/cleaned

2. Prefetch images (optional but recommended)
   python multimodal_model.py \
       --csv data/cleaned/all_splits.csv \
       --prefetch-images

3. Run full multimodal training
   python multimodal_model.py \
       --csv data/cleaned/all_splits.csv \
       --epochs 30 \
       --batch-size 32 \
       --out-dir outputs/multimodal

Arguments (common):
--csv                Path to CSV containing description, image URL, label
--image-cache        Directory for cached images
--prefetch-images    Download all unique URLs before training
--prefetch-timeout   Per-request timeout for async downloader
--prefetch-concurrency  Number of concurrent image download workers
--batch-size         Batch size for training
--epochs             Training epochs
--device             cuda/mps/cpu (auto-detect if omitted)

Outputs:
- outputs/results.json
- outputs/results.csv
- outputs/test_preds_<model>.npz
- outputs/embeddings_<split>_<model>.pt
- outputs/<model>.pth
"""

import os
import json
import time
import argparse
import hashlib
import warnings
import asyncio
import io
from typing import Dict, Any, Tuple, List

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from transformers import (
    AutoTokenizer,
    AutoModel,
    CLIPVisionModel,
    CLIPImageProcessor,
)

import aiohttp
from aiohttp import ClientTimeout, TCPConnector

# -------------------------------------------------------
# Image cache utilities and async prefetch
# -------------------------------------------------------

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BotDetector/1.0; +https://example.org/botdetector)"
}

def url_to_cache_path(url: str, cache_dir: str) -> str:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, h + ".jpg")


def save_bytes_to_jpeg(data: bytes, path: str) -> bool:
    """Decode bytes as RGB image and save as JPEG. Returns True if successful."""
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.save(path, format="JPEG", quality=85)
        return True
    except Exception:
        return False


def log_failed_url(url: str, cache_dir: str):
    """Append a failed URL to failed_urls.txt for debugging."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, "failed_urls.txt"), "a", encoding="utf-8") as f:
            f.write(f"{url}\n")
    except Exception:
        pass


async def _fetch_and_cache(
    session: aiohttp.ClientSession,
    url: str,
    cache_path: str,
    max_bytes: int,
    sem: asyncio.Semaphore,
    timeout: float,
):
    """
    Single async fetch + save to cache_path.
    Returns (url, True/False, message).
    """
    # Skip if already cached
    if os.path.exists(cache_path):
        return (url, True, "cached")

    async with sem:
        try:
            async with session.get(url, timeout=ClientTimeout(total=timeout)) as resp:
                if resp.status != 200:
                    msg = f"HTTP {resp.status}"
                    log_failed_url(url, os.path.dirname(cache_path))
                    return (url, False, msg)

                total = 0
                chunks = []
                async for chunk in resp.content.iter_chunked(8192):
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        msg = f"TooLarge({total})"
                        log_failed_url(url, os.path.dirname(cache_path))
                        return (url, False, msg)
                    chunks.append(chunk)

                data = b"".join(chunks)
                ok = save_bytes_to_jpeg(data, cache_path)
                if not ok:
                    msg = "decode-failed"
                    log_failed_url(url, os.path.dirname(cache_path))
                    return (url, False, msg)

                return (url, True, "ok")
        except Exception as e:
            msg = repr(e)
            log_failed_url(url, os.path.dirname(cache_path))
            return (url, False, msg)


async def async_prefetch_image_cache(
    df: pd.DataFrame,
    image_col: str,
    cache_dir: str,
    max_workers: int = 32,
    max_bytes: int = 5_000_000,
    timeout: float = 10.0,
):
    """
    Prefetch unique HTTP(S) URLs in df[image_col] concurrently using aiohttp.
    No synchronous downloads will be done elsewhere; training will rely entirely on this cache.
    """
    os.makedirs(cache_dir, exist_ok=True)

    urls = pd.Series(df[image_col].dropna().unique()).astype(str).tolist()
    urls = [u for u in urls if u.lower().startswith(("http://", "https://"))]

    if len(urls) == 0:
        print("[*] No URLs to prefetch.")
        return

    connector = TCPConnector(limit_per_host=max_workers, ssl=False)
    sem = asyncio.Semaphore(max_workers)

    successes = 0
    failures = 0
    tasks = []

    async with aiohttp.ClientSession(
        headers=DEFAULT_HEADERS,
        connector=connector,
        timeout=ClientTimeout(total=None),
    ) as session:
        for u in urls:
            cache_path = url_to_cache_path(u, cache_dir)
            tasks.append(_fetch_and_cache(session, u, cache_path, max_bytes, sem, timeout))

        print(f"[+] Async prefetch starting: {len(tasks)} HTTP downloads (concurrency={max_workers}) ...")

        BATCH = 256
        done = 0

        for i in range(0, len(tasks), BATCH):
            batch = tasks[i:i + BATCH]
            results = await asyncio.gather(*batch)
            for (url, ok, msg) in results:
                done += 1
                if ok:
                    successes += 1
                else:
                    failures += 1
                if done % 250 == 0 or done == len(tasks):
                    print(f"   prefetched {done}/{len(tasks)} (succ={successes} fail={failures})")

    print(f"[+] Async prefetch done. success={successes} fail={failures}")


def fetch_image_from_cache_or_placeholder(ref: str, cache_dir: str) -> Image.Image:
    """
    Return PIL image if cached/exists; otherwise a black placeholder (224x224).
    No network I/O here – we rely solely on the prefetch step.
    """
    try:
        # Remote URL: look in cache
        if isinstance(ref, str) and ref.lower().startswith(("http://", "https://")):
            p = url_to_cache_path(ref, cache_dir)
            if os.path.exists(p):
                try:
                    return Image.open(p).convert("RGB")
                except Exception:
                    # corrupted cache file; remove and mark as failed
                    try:
                        os.remove(p)
                    except Exception:
                        pass
                    log_failed_url(ref, cache_dir)
                    return Image.new("RGB", (224, 224), (0, 0, 0))

            # not cached
            log_failed_url(ref, cache_dir)
            return Image.new("RGB", (224, 224), (0, 0, 0))

        # Local path
        if os.path.exists(str(ref)):
            try:
                return Image.open(str(ref)).convert("RGB")
            except Exception:
                log_failed_url(str(ref), cache_dir)
                return Image.new("RGB", (224, 224), (0, 0, 0))

        # Missing local file
        log_failed_url(str(ref), cache_dir)
        return Image.new("RGB", (224, 224), (0, 0, 0))
    except Exception:
        log_failed_url(str(ref), cache_dir)
        return Image.new("RGB", (224, 224), (0, 0, 0))


def exists_in_cache_or_fs(ref: str, cache_dir: str) -> int:
    """
    Return 0 if image present (either cached URL or existing local path), 1 if missing.
    This is used to set the image_missing flag.
    """
    try:
        if isinstance(ref, str) and ref.lower().startswith(("http://", "https://")):
            return 0 if os.path.exists(url_to_cache_path(ref, cache_dir)) else 1
        return 0 if os.path.exists(str(ref)) else 1
    except Exception:
        return 1


# -------------------------------------------------------
# Dataset
# -------------------------------------------------------

class BotDetectionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_col: str,
        bio_col: str,
        label_col: str,
        image_cache: str,
    ):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.bio_col = bio_col
        self.label_col = label_col
        self.image_cache = image_cache

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        bio = r[self.bio_col]
        if pd.isna(bio) or str(bio).strip() == "":
            bio = "<NO_BIO>"
        img_ref = r[self.image_col]
        label = int(r[self.label_col])

        image_missing = exists_in_cache_or_fs(img_ref, self.image_cache)
        image = fetch_image_from_cache_or_placeholder(img_ref, self.image_cache)

        return {
            "bio": bio,
            "image": image,
            "label": label,
            "has_bio": 0 if bio == "<NO_BIO>" else 1,
            "image_missing": int(image_missing),
            "img_ref": img_ref,
        }


def collate_batch(
    batch: List[Dict[str, Any]],
    tokenizer,
    image_processor,
    max_length: int,
) -> Dict[str, Any]:
    """Keep EVERYTHING on CPU here. Move to GPU later inside the loops."""
    images = [b["image"] for b in batch]
    bios = [b["bio"] for b in batch]

    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)          # CPU
    has_bio = torch.tensor([b["has_bio"] for b in batch], dtype=torch.float32)    # CPU
    image_missing = torch.tensor([b["image_missing"] for b in batch], dtype=torch.float32)  # CPU

    pixel_values = image_processor(images=images, return_tensors="pt")["pixel_values"]  # CPU
    tokenized = tokenizer(
        bios, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )  # CPU

    return {
        "pixel_values": pixel_values,                    # (B, 3, H, W) CPU
        "input_ids": tokenized["input_ids"],             # CPU
        "attention_mask": tokenized["attention_mask"],   # CPU
        "labels": labels,                                # CPU
        "has_bio": has_bio,                              # CPU
        "image_missing": image_missing,                  # CPU
    }


# -------------------------------------------------------
# Backbones
# -------------------------------------------------------

def get_text_backbone(device: str):
    text_name = "distilroberta-base"
    tok = AutoTokenizer.from_pretrained(
        text_name
        )
    model = AutoModel.from_pretrained(
        text_name, 
        ).to(device)
    hidden = model.config.hidden_size  # 768
    return tok, model, hidden


def get_vision_backbone(device: str):
    vis_name = "openai/clip-vit-base-patch32"
    image_processor = CLIPImageProcessor.from_pretrained(
        vis_name, 
        )
    vision_model = CLIPVisionModel.from_pretrained(
        vis_name,
        # use_safetensors=True,
        # dtype=torch.float32,   # avoid torch_dtype deprecation warnings
    ).to(device)
    hidden = vision_model.config.hidden_size  # 768
    return image_processor, vision_model, hidden


# -------------------------------------------------------
# Heads / Multimodal models
# -------------------------------------------------------

class TextOnly(nn.Module):
    def __init__(self, text_model: AutoModel, hidden: int = 768):
        super().__init__()
        self.text = text_model
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask, **_):
        out = self.text(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [CLS]/first token
        return self.classifier(cls)


class ImageOnly(nn.Module):
    def __init__(self, vision_model: CLIPVisionModel, hidden: int = 768):
        super().__init__()
        self.vision = vision_model
        self.missing_img_emb = nn.Parameter(torch.randn(hidden) * 0.02)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, pixel_values, image_missing, **_):
        vis_out = self.vision(pixel_values=pixel_values).last_hidden_state
        img_cls = vis_out[:, 0, :]  # (B, 768)
        miss = image_missing.unsqueeze(1)  # (B,1)
        missing_vec = self.missing_img_emb.unsqueeze(0)  # (1,768)
        img_replaced = img_cls * (1.0 - miss) + missing_vec * miss
        return self.classifier(img_replaced)


class ConcatFusion(nn.Module):
    def __init__(self, text_model: AutoModel, vision_model: CLIPVisionModel, hidden: int = 768):
        super().__init__()
        self.text = text_model
        self.vision = vision_model
        self.missing_img_emb = nn.Parameter(torch.randn(hidden) * 0.02)
        self.fuse = nn.Linear(hidden * 2, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask, pixel_values, image_missing, **_):
        t = self.text(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        v_tokens = self.vision(pixel_values=pixel_values).last_hidden_state
        v = v_tokens[:, 0, :]
        miss = image_missing.unsqueeze(1)
        v = v * (1.0 - miss) + self.missing_img_emb.unsqueeze(0) * miss
        fused = self.fuse(torch.cat([t, v], dim=1))
        return self.classifier(fused)


class GMUFusion(nn.Module):
    def __init__(self, text_model: AutoModel, vision_model: CLIPVisionModel, hidden: int = 768):
        super().__init__()
        self.text = text_model
        self.vision = vision_model
        self.missing_img_emb = nn.Parameter(torch.randn(hidden) * 0.02)
        self.text_proj = nn.Linear(hidden, hidden)
        self.image_proj = nn.Linear(hidden, hidden)
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask, pixel_values, image_missing, **_):
        t_raw = self.text(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        v_tokens = self.vision(pixel_values=pixel_values).last_hidden_state
        v_raw = v_tokens[:, 0, :]
        miss = image_missing.unsqueeze(1)
        v_raw = v_raw * (1.0 - miss) + self.missing_img_emb.unsqueeze(0) * miss

        t = torch.tanh(self.text_proj(t_raw))
        v = torch.tanh(self.image_proj(v_raw))
        z = self.gate(torch.cat([t_raw, v_raw], dim=1))
        h = z * t + (1 - z) * v
        return self.classifier(h)


class CrossAttentionFusion(nn.Module):
    def __init__(self, text_model: AutoModel, vision_model: CLIPVisionModel, hidden: int = 768, num_heads: int = 8):
        super().__init__()
        self.text = text_model
        self.vision = vision_model
        self.missing_img_emb = nn.Parameter(torch.randn(hidden) * 0.02)

        self.text_proj = nn.Linear(hidden, hidden)
        self.image_proj = nn.Linear(hidden, hidden)

        self.t2i = nn.MultiheadAttention(embed_dim=hidden, num_heads=num_heads, batch_first=True)
        self.i2t = nn.MultiheadAttention(embed_dim=hidden, num_heads=num_heads, batch_first=True)

        self.fuse = nn.Linear(hidden * 2, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask, pixel_values, image_missing, **_):
        t_seq = self.text(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (B,Lt,768)
        i_seq = self.vision(pixel_values=pixel_values).last_hidden_state  # (B,Li,768)

        miss = image_missing.view(-1, 1, 1)  # (B,1,1)
        missing_tok = self.missing_img_emb.view(1, 1, -1)  # (1,1,768)
        i_seq = i_seq * (1.0 - miss) + missing_tok * miss

        t_proj = self.text_proj(t_seq)
        i_proj = self.image_proj(i_seq)

        t_att, _ = self.t2i(t_proj, i_proj, i_proj)
        i_query = i_proj.mean(dim=1, keepdim=True)
        i_att, _ = self.i2t(i_query, t_proj, t_proj)

        t_pool = t_att[:, 0, :]
        i_pool = i_att[:, 0, :]

        fused = self.fuse(torch.cat([t_pool, i_pool], dim=1))
        return self.classifier(fused)


# -------------------------------------------------------
# Metrics / eval helpers
# -------------------------------------------------------

def compute_metrics_from_logits(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, Any]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = float("nan")
    try:
        pr = average_precision_score(y_true, probs)
    except Exception:
        pr = float("nan")
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "auc": float(auc),
        "pr": float(pr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "y_pred": preds,
        "y_prob": probs,
    }


@torch.no_grad()
def evaluate_dl(model, loader, device) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_list = []
    labels_list = []
    for batch in tqdm(loader, desc="Evaluating"):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        logits = model(
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            pixel_values=batch.get("pixel_values"),
            image_missing=batch.get("image_missing"),
            has_bio=batch.get("has_bio"),
        )
        labels_list.append(batch["labels"].cpu().numpy())
        if logits.ndim == 2 and logits.shape[1] == 2:
            logit_pos = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()
            logits_list.append(logit_pos)
        else:
            logits_list.append(logits.detach().cpu().numpy())
    logits = np.concatenate(logits_list, axis=0)
    y_true = np.concatenate(labels_list, axis=0)
    metrics = compute_metrics_from_logits(y_true, logits)
    return metrics, y_true, metrics["y_prob"], metrics["y_pred"]


def export_preds(out_dir: str, model_key: str, y_true, y_prob, y_pred):
    np.savez(os.path.join(out_dir, f"test_preds_{model_key}.npz"),
             y_true=y_true, y_prob=y_prob, y_pred=y_pred)


def export_embeddings(out_dir: str, split: str, model_key: str,
                      image_embs: np.ndarray, text_embs: np.ndarray,
                      labels: np.ndarray, has_bio: np.ndarray, image_missing: np.ndarray):
    torch.save({
        "image_embs": image_embs,
        "text_embs": text_embs,
        "labels": labels,
        "has_bio": has_bio,
        "image_missing": image_missing,
    }, os.path.join(out_dir, f"embeddings_{split}_{model_key}.pt"))


# -------------------------------------------------------
# Embedding extraction
# -------------------------------------------------------

@torch.no_grad()
def extract_simple_embeddings(
    loader: DataLoader,
    text_model: AutoModel,
    vision_model: CLIPVisionModel,
    device: str
):
    text_model.eval()
    vision_model.eval()
    image_embs = []
    text_embs = []
    labels = []
    has_bio = []
    image_missing = []

    for batch in tqdm(loader, desc="Embeddings"):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        t_out = text_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                           output_hidden_states=False, return_dict=True)
        last_hidden = t_out.last_hidden_state  # (B,L,768)
        att_mask = batch["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)  # (B,L,1)
        pooled = (last_hidden * att_mask).sum(dim=1) / att_mask.sum(dim=1).clamp(min=1.0)  # (B,768)
        text_embs.append(pooled.detach().cpu())

        v_tokens = vision_model(pixel_values=batch["pixel_values"]).last_hidden_state  # (B,Li,768)
        v_cls = v_tokens[:, 0, :]  # (B,768)
        image_embs.append(v_cls.detach().cpu())

        labels.append(batch["labels"].cpu())
        has_bio.append(batch["has_bio"].cpu())
        image_missing.append(batch["image_missing"].cpu())

    image_embs = torch.cat(image_embs, dim=0).numpy()
    text_embs = torch.cat(text_embs, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    has_bio = torch.cat(has_bio, dim=0).numpy()
    image_missing = torch.cat(image_missing, dim=0).numpy()
    return image_embs, text_embs, labels, has_bio, image_missing


# -------------------------------------------------------
# Training loop with curves
# -------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    model_key: str
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_val_auc = -1.0
    curves = {"epoch": [], "loss": [], "val_auc": []}

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in tqdm(train_loader, desc=f"Training [{model_key}]"):
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            logits = model(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                pixel_values=batch.get("pixel_values"),
                image_missing=batch.get("image_missing"),
                has_bio=batch.get("has_bio"),
            )
            loss = criterion(logits, batch["labels"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * logits.size(0)
            n += logits.size(0)

        avg_loss = total_loss / max(1, n)
        val_metrics, _, _, _ = evaluate_dl(model, val_loader, device)
        val_auc = val_metrics["auc"]

        curves["epoch"].append(ep)
        curves["loss"].append(float(avg_loss))
        curves["val_auc"].append(float(val_auc))

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, curves


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Multimodal Bot Detection with CLIP (vision) + DistilRoBERTa (text)")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--image-col", default="profile_image_url")
    ap.add_argument("--bio-col", default="description")
    ap.add_argument("--label-col", default="account_type")
    ap.add_argument("--image-cache", default="data/image_cache")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)  # cuda/mps/cpu
    ap.add_argument("--max-length", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--skip-text-only", action="store_true")
    ap.add_argument("--skip-image-only", action="store_true")
    ap.add_argument("--prefetch-images", action="store_true",
                    help="Asynchronously download & cache all images before training")
    ap.add_argument("--prefetch-concurrency", type=int, default=32,
                    help="Max concurrent HTTP requests for prefetch")
    ap.add_argument("--prefetch-timeout", type=float, default=6.0,
                    help="Timeout per request during prefetch")
    args = ap.parse_args()

    # Device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.image_cache, exist_ok=True)

    # Load CSV
    df = pd.read_csv(args.csv)
    if args.label_col not in df.columns or args.image_col not in df.columns or args.bio_col not in df.columns:
        raise ValueError("CSV must contain the specified --label-col, --image-col, and --bio-col")

    # Map labels if strings
    if df[args.label_col].dtype == object or df[args.label_col].dtype == "O":
        df[args.label_col] = df[args.label_col].map({"bot": 1, "human": 0})

    # Optional async prefetch over the full dataset
    if args.prefetch_images:
        print("[*] Starting async prefetch. This may take time depending on number of URLs.")
        asyncio.run(
            async_prefetch_image_cache(
                df=df,
                image_col=args.image_col,
                cache_dir=args.image_cache,
                max_workers=args.prefetch_concurrency,
                max_bytes=5_000_000,
                timeout=args.prefetch_timeout,
            )
        )
    else:
        print("[*] Skipping image prefetch. Will rely on whatever is already in the cache/local FS.")

    # Split
    if "split" in df.columns:
        train_df = df[df["split"] == "train"].drop(columns=["split"])
        val_df = df[df["split"] == "val"].drop(columns=["split"])
        test_df = df[df["split"] == "test"].drop(columns=["split"])
    else:
        trainval_df, test_df = train_test_split(
            df,
            test_size=0.10,
            random_state=args.seed,
            stratify=df[args.label_col],
        )
        train_df, val_df = train_test_split(
            trainval_df,
            test_size=0.20,
            random_state=args.seed,
            stratify=trainval_df[args.label_col],
        )

    print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Backbones
    tokenizer, text_model, text_hidden = get_text_backbone(device)
    image_processor, vision_model, vision_hidden = get_vision_backbone(device)
    assert text_hidden == vision_hidden == 768, "Expected 768-dim hidden for both backbones."

    # Datasets / Loaders
    def make_loader(split_df, shuffle: bool):
        ds = BotDetectionDataset(split_df, args.image_col, args.bio_col, args.label_col, args.image_cache)
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=(device == "cuda"),
            collate_fn=lambda b: collate_batch(b, tokenizer, image_processor, args.max_length),
        )

    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df, shuffle=False)
    test_loader = make_loader(test_df, shuffle=False)

    # Fresh backbones for each experiment
    def fresh_text():
        return AutoModel.from_pretrained("distilroberta-base").to(device)

    def fresh_vision():
        return CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            # use_safetensors=True,
            # dtype=torch.float32
        ).to(device)

    experiments = []
    if not args.skip_text_only:
        experiments.append(("text_only", TextOnly(fresh_text(), 768)))
    if not args.skip_image_only:
        experiments.append(("image_only", ImageOnly(fresh_vision(), 768)))
    experiments.extend([
        ("concat",          ConcatFusion(fresh_text(), fresh_vision(), 768)),
        ("gmu",             GMUFusion(fresh_text(), fresh_vision(), 768)),
        ("cross_attention", CrossAttentionFusion(fresh_text(), fresh_vision(), 768, num_heads=8)),
    ])

    combined_results_json: Dict[str, Any] = {}
    combined_rows_csv: List[Dict[str, Any]] = []

    # Run experiments
    for model_key, model in experiments:
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {model_key}")
        print("=" * 80)

        # Train
        model = model.to(device)
        model, curves = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            model_key=model_key
        )

        # Evaluate
        val_metrics, _, _, _ = evaluate_dl(model, val_loader, device)
        test_metrics, y_true, y_prob, y_pred = evaluate_dl(model, test_loader, device)

        # Export predictions
        export_preds(args.out_dir, model_key, y_true, y_prob, y_pred)

        # Embeddings (use trained backbones if available)
        t_backbone = getattr(model, "text", fresh_text())
        v_backbone = getattr(model, "vision", fresh_vision())
        for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
            img_emb, txt_emb, lbls, hb, im = extract_simple_embeddings(loader, t_backbone, v_backbone, device)
            export_embeddings(args.out_dir, split_name, model_key, img_emb, txt_emb, lbls, hb, im)

        # Record results
        combined_results_json[model_key] = {
            "val": {
                "auc": val_metrics["auc"], "pr": val_metrics["pr"],
                "precision": val_metrics["precision"], "recall": val_metrics["recall"],
                "f1": val_metrics["f1"], "tp": val_metrics["tp"], "fp": val_metrics["fp"],
                "tn": val_metrics["tn"], "fn": val_metrics["fn"],
            },
            "test": {
                "auc": test_metrics["auc"], "pr": test_metrics["pr"],
                "precision": test_metrics["precision"], "recall": test_metrics["recall"],
                "f1": test_metrics["f1"], "tp": test_metrics["tp"], "fp": test_metrics["fp"],
                "tn": test_metrics["tn"], "fn": test_metrics["fn"],
            },
            "training_curve": curves
        }

        combined_rows_csv.append({
            "model_type": model_key,
            "val_auc": val_metrics["auc"],
            "val_pr": val_metrics["pr"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "test_auc": test_metrics["auc"],
            "test_pr": test_metrics["pr"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "tp": test_metrics["tp"],
            "fp": test_metrics["fp"],
            "tn": test_metrics["tn"],
            "fn": test_metrics["fn"],
        })

        # Save model weights
        torch.save(model.state_dict(), os.path.join(args.out_dir, f"{model_key}.pth"))

    # Write combined results
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(combined_results_json, f, indent=2)

    pd.DataFrame(combined_rows_csv).to_csv(
        os.path.join(args.out_dir, "results.csv"),
        index=False,
        float_format="%.6f"
    )

    print("\n[+] All experiments completed. Combined results written to results.json and results.csv")


if __name__ == "__main__":
    main()