#!/usr/bin/env python3
"""
train_multimodal_jh.py

[+] Async prefetch done. success=18879 fail=17685
1120 start with prefetched done.12 still not done 

Usage example:
python src/model/train_multimodal_jh.py \
  --csv data/cleaned/twitter_human_bots_dataset_cleaned.csv \
  --image-col profile_image_url \
  --bio-col description \
  --label-col account_type \
  --out-dir outputs/ \
  --batch-size 32 \
  --epochs 30 \
  --prefetch-concurrency 32 \
  --save-embeddings \
  --prefetch-images 

Updated:
- logs failed URLs to image_cache/failed_urls.txt
- sets image_missing flag when cache not present (or local file missing)
- includes image_missing in embeddings output
- uses a learnable missing-image embedding inside the classifier (replaces black-placeholder behavior)
"""

import os
import argparse
import random
import hashlib
import io
from pathlib import Path
from typing import Dict, Any
import asyncio
import aiohttp  # async HTTP client
from aiohttp import ClientTimeout, TCPConnector
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import chi2_contingency

import requests  # still used for single sync fetch fallback
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from transformers import (
    CLIPModel,
    CLIPProcessor,
    AutoTokenizer,
    AutoModel,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# ------------------------------
# Image caching utilities (sha1 naming)
# ------------------------------
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BotDetector/1.0; +https://example.org/botdetector)"
}

def url_to_cache_path(url: str, cache_dir: str) -> str:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    ext = ".jpg"
    return os.path.join(cache_dir, h + ext)

def save_bytes_to_jpeg(data: bytes, path: str):
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        # write with reasonable JPEG compression
        img.save(path, format="JPEG", quality=85)
        return True
    except Exception:
        # If decoding fails, return False
        return False

# ------------------------------
# Async prefetch implementation (aiohttp)
# ------------------------------
async def _fetch_and_cache(session: aiohttp.ClientSession, url: str, cache_path: str,
                           max_bytes: int, sem: asyncio.Semaphore, timeout: float):
    """Single async fetch + save to cache_path. Returns (url, True/False, errmsg)."""
    # quick skip if already cached
    if os.path.exists(cache_path):
        return (url, True, "cached")
    async with sem:
        try:
            # per-request timeout
            async with session.get(url, timeout=ClientTimeout(total=timeout)) as resp:
                if resp.status != 200:
                    text = await resp.text(errors="ignore")
                    return (url, False, f"HTTP {resp.status}")
                total = 0
                chunks = []
                async for chunk in resp.content.iter_chunked(8192):
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        return (url, False, f"TooLarge({total})")
                    chunks.append(chunk)
                data = b"".join(chunks)
                ok = save_bytes_to_jpeg(data, cache_path)
                if not ok:
                    return (url, False, "decode-failed")
                return (url, True, "ok")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return (url, False, repr(e))

async def async_prefetch_image_cache(df: pd.DataFrame, image_col: str, cache_dir: str,
                                     max_workers: int = 32, max_bytes: int = 5_000_000,
                                     timeout: float = 10.0):
    """Prefetch unique URLs in df[image_col] concurrently using aiohttp."""
    os.makedirs(cache_dir, exist_ok=True)
    urls = pd.Series(df[image_col].dropna().unique()).astype(str).tolist()
    total = len(urls)
    if total == 0:
        print("[*] No URLs to prefetch.")
        return

    connector = TCPConnector(limit_per_host=max_workers, ssl=False)
    timeout_cfg = ClientTimeout(total=None)  # we'll use per-request timeouts
    sem = asyncio.Semaphore(max_workers)
    successes = 0
    failures = 0
    async with aiohttp.ClientSession(headers=DEFAULT_HEADERS, connector=connector, timeout=timeout_cfg) as session:
        tasks = []
        for u in urls:
            if not str(u).lower().startswith(("http://", "https://")):
                # skip local paths in async prefetch
                continue
            cache_path = url_to_cache_path(u, cache_dir)
            # schedule fetch
            tasks.append(_fetch_and_cache(session, u, cache_path, max_bytes, sem, timeout))

        # run with progress reporting
        print(f"[+] Async prefetch starting: {len(tasks)} HTTP downloads (concurrency={max_workers}) ...")
        # gather in batches to show progress and avoid exploding memory for HUGE lists
        done = 0
        BATCH = 256  # process gather in chunks to reduce spike memory
        for i in range(0, len(tasks), BATCH):
            batch = tasks[i:i+BATCH]
            results = await asyncio.gather(*batch, return_exceptions=False)
            for (url, ok, msg) in results:
                done += 1
                if ok:
                    successes += 1
                else:
                    failures += 1
                    # log failures to a file so we can inspect later
                    try:
                        with open(os.path.join(cache_dir, "failed_urls.txt"), "a", encoding="utf-8") as f:
                            f.write(f"{url}\t{msg}\n")
                    except Exception:
                        pass
                    if done % 50 == 0:
                        print(f"   [prefetch] failed {failures} so far; last: {url} -> {msg}")
            if done % 100 == 0 or done == len(tasks):
                print(f"   prefetched {done}/{len(tasks)} (succ={successes} fail={failures})")
    print(f"[+] Async prefetch done. success={successes} fail={failures}")

# ------------------------------
# Fallback synchronous downloader
# ------------------------------
def download_image_to_pil_sync(url: str, timeout: float = 6.0, max_bytes: int = 5_000_000) -> Image.Image:
    with requests.get(url, headers=DEFAULT_HEADERS, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = 0
        chunks = []
        for chunk in r.iter_content(chunk_size=8192):
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"Image too large (> {max_bytes} bytes): {url}")
            chunks.append(chunk)
        data = b"".join(chunks)
        bio = io.BytesIO(data)
        img = Image.open(bio).convert("RGB")
        return img

def log_failed_url(url: str, cache_dir: str):
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, "failed_urls.txt"), "a", encoding="utf-8") as f:
            f.write(f"{url}\n")
    except Exception:
        pass

def fetch_image_with_cache(path_or_url: str, cache_dir: str, retries: int = 1, timeout: float = 6.0) -> Image.Image:
    """
    If path_or_url is a URL, attempt to load from cache or download and cache.
    If it's a local path, open directly.
    On failure return a black placeholder image (224x224).
    NOTE: this function intentionally does NOT try to download when cache missing -
    we assume you used prefetch. It will return placeholder immediately and we set image_missing accordingly in Dataset.
    """
    os.makedirs(cache_dir, exist_ok=True)
    try:
        if isinstance(path_or_url, str) and path_or_url.lower().startswith(("http://", "https://")):
            cache_path = url_to_cache_path(path_or_url, cache_dir)
            if os.path.exists(cache_path):
                try:
                    return Image.open(cache_path).convert("RGB")
                except Exception:
                    try:
                        os.remove(cache_path)
                    except Exception:
                        pass
                    log_failed_url(path_or_url, cache_dir)
                    return Image.new("RGB", (224, 224), (0, 0, 0))
            # Cache missing -> we do not attempt synchronous download here; treat as missing
            log_failed_url(path_or_url, cache_dir)
            return Image.new("RGB", (224, 224), (0, 0, 0))
        else:
            if not os.path.exists(path_or_url):
                log_failed_url(path_or_url, cache_dir)
                return Image.new("RGB", (224, 224), (0, 0, 0))
            return Image.open(path_or_url).convert("RGB")
    except Exception as e:
        print(f"[!] Image load failed for {path_or_url!r}: {repr(e)} -- using placeholder")
        log_failed_url(path_or_url, cache_dir)
        return Image.new("RGB", (224, 224), (0, 0, 0))

# ------------------------------
# Dataset that uses URL/local path loader and exposes image_missing flag
# ------------------------------
class ProfileImageTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_col: str, bio_col: str, label_col: str,
                 clip_processor: CLIPProcessor, tokenizer, max_length: int = 64,
                 image_cache: str = "image_cache"):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.bio_col = bio_col
        self.label_col = label_col
        self.clip_processor = clip_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_cache = image_cache

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_ref = row[self.image_col]
        bio = row[self.bio_col] if pd.notna(row[self.bio_col]) and str(row[self.bio_col]).strip() != "" else "<NO_BIO>"
        label = int(row[self.label_col])
        has_bio = 0 if bio == "<NO_BIO>" else 1

        # Determine image_missing by cache existence (for URLs) or file existence (for local)
        image_missing = 0
        if isinstance(img_ref, str) and img_ref.lower().startswith(("http://", "https://")):
            cache_path = url_to_cache_path(img_ref, self.image_cache)
            if not os.path.exists(cache_path):
                image_missing = 1
                image = Image.new("RGB", (224, 224), (0, 0, 0))
                # log already handled by fetch_image_with_cache/prefetch; ensure it's present in failed file
                try:
                    with open(os.path.join(self.image_cache, "failed_urls.txt"), "a", encoding="utf-8") as f:
                        f.write(f"{img_ref}\n")
                except Exception:
                    pass
            else:
                # load cached image
                try:
                    image = Image.open(cache_path).convert("RGB")
                except Exception:
                    image_missing = 1
                    image = Image.new("RGB", (224, 224), (0, 0, 0))
                    log_failed_url(img_ref, self.image_cache)
        else:
            # local path handling
            if not os.path.exists(str(img_ref)):
                image_missing = 1
                image = Image.new("RGB", (224, 224), (0, 0, 0))
                log_failed_url(str(img_ref), self.image_cache)
            else:
                try:
                    image = Image.open(str(img_ref)).convert("RGB")
                except Exception:
                    image_missing = 1
                    image = Image.new("RGB", (224, 224), (0, 0, 0))
                    log_failed_url(str(img_ref), self.image_cache)

        return {
            "image": image,
            "bio": bio,
            "label": label,
            "has_bio": has_bio,
            "image_missing": int(image_missing),
            "img_ref": img_ref,
        }

def collate_for_embedding(batch, clip_processor, tokenizer, max_length=64, device="cpu"):
    images = [b["image"] for b in batch]
    bios = [b["bio"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long, device=device)
    has_bio = torch.tensor([b["has_bio"] for b in batch], dtype=torch.float32, device=device)
    image_missing = torch.tensor([b["image_missing"] for b in batch], dtype=torch.float32, device=device)

    clip_inputs = clip_processor(images=images, return_tensors="pt")
    tokenized = tokenizer(bios, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    return {
        "clip_inputs": {k: v.to(device) for k, v in clip_inputs.items()},
        "tokenized": {k: v.to(device) for k, v in tokenized.items()},
        "labels": labels,
        "has_bio": has_bio,
        "image_missing": image_missing,
    }

# ------------------------------
# MLP classifier with learnable missing-image embedding
# ------------------------------
class MLPClassifierWithMissing(nn.Module):
    """
    Expects input composed of: [img_emb, txt_emb, has_bio, image_missing]
    During forward, replaces img_emb for samples with image_missing==1 by a learnable vector.
    Then concatenates [img_replaced, txt_emb, has_bio] and runs through MLP.
    """
    def __init__(self, image_emb_dim: int, text_emb_dim: int, hidden_dims=(1024, 256), dropout=0.3):
        super().__init__()
        self.image_emb_dim = image_emb_dim
        self.text_emb_dim = text_emb_dim
        self.in_dim = image_emb_dim + text_emb_dim + 1  # has_bio included
        # learnable missing embedding:
        self.missing_img_emb = nn.Parameter(torch.randn(image_emb_dim) * 0.02)
        # classifier MLP
        layers = []
        prev = self.in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (B, image_dim + text_dim + 2)  (last two: has_bio, image_missing)
        """
        img = x[:, : self.image_emb_dim]  # (B, image_dim)
        txt = x[:, self.image_emb_dim : self.image_emb_dim + self.text_emb_dim]  # (B, text_dim)
        has_bio = x[:, -2].unsqueeze(1)  # (B,1)
        image_missing = x[:, -1].unsqueeze(1)  # (B,1) values 0/1

        # replace image embedding where missing==1 with learnable vector
        # broadcast missing emb
        missing_vec = self.missing_img_emb.unsqueeze(0)  # (1, image_dim)
        img_replaced = img * (1.0 - image_missing) + missing_vec * image_missing

        joint = torch.cat([img_replaced, txt, has_bio], dim=1)  # (B, in_dim)
        logits = self.net(joint).squeeze(-1)
        return logits

# ------------------------------
# Embedding extraction (adds image_missing to outputs)
# ------------------------------
@torch.no_grad()
def extract_embeddings(
    df: pd.DataFrame,
    image_col: str,
    bio_col: str,
    label_col: str,
    clip_model,
    clip_processor,
    text_model,
    tokenizer,
    device: str,
    batch_size: int = 32,
    max_length: int = 64,
    image_cache: str = "image_cache",
    save_path: str = None,
) -> Dict:
    
    ds = ProfileImageTextDataset(df, image_col, bio_col, label_col, clip_processor, tokenizer, max_length=max_length, image_cache=image_cache)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_for_embedding(b, clip_processor, tokenizer, max_length, device))

    clip_model.eval()
    text_model.eval()

    image_embs = []
    text_embs = []
    labels = []
    has_bios = []
    image_missing_list = []

    for batch in tqdm(loader):
        clip_inputs = batch["clip_inputs"]
        tokenized = batch["tokenized"]

        clip_outputs = clip_model.get_image_features(**clip_inputs)
        image_emb = clip_outputs.detach().cpu()
        image_embs.append(image_emb)

        text_outputs = text_model(**tokenized, output_hidden_states=False, return_dict=True)
        last_hidden = text_outputs.last_hidden_state
        att_mask = tokenized["attention_mask"].unsqueeze(-1)
        att_mask = att_mask.to(last_hidden.dtype)
        summed = (last_hidden * att_mask).sum(dim=1)
        lens = att_mask.sum(dim=1).clamp(min=1.0)
        pooled = (summed / lens)
        text_emb = pooled.detach().cpu()
        text_embs.append(text_emb)

        labels.append(batch["labels"].cpu())
        has_bios.append(batch["has_bio"].cpu())
        image_missing_list.append(batch["image_missing"].cpu())

    image_embs = torch.cat(image_embs, dim=0)
    text_embs = torch.cat(text_embs, dim=0)
    labels = torch.cat(labels, dim=0)
    has_bios = torch.cat(has_bios, dim=0)
    image_missing_arr = torch.cat(image_missing_list, dim=0)

    output = {
        "image_embs": image_embs.numpy(),
        "text_embs": text_embs.numpy(),
        "labels": labels.numpy(),
        "has_bio": has_bios.numpy(),
        "image_missing": image_missing_arr.numpy(),
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(output, save_path)
        print(f"[+] Saved embeddings to {save_path}")

    return output

# ------------------------------
# Train classifier
# ------------------------------
def train_classifier(
    train_embeddings: Dict[str, np.ndarray],
    val_embeddings: Dict[str, np.ndarray],
    test_embeddings: Dict[str, np.ndarray],
    out_dir: str,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-4,
    modality_dropout: float = 0.2,
    device: str = "cpu",
):
    X_img_tr = train_embeddings["image_embs"]
    X_txt_tr = train_embeddings["text_embs"]
    y_tr = train_embeddings["labels"]
    has_tr = train_embeddings["has_bio"]
    img_miss_tr = train_embeddings["image_missing"]

    X_img_val = val_embeddings["image_embs"]
    X_txt_val = val_embeddings["text_embs"]
    y_val = val_embeddings["labels"]
    has_val = val_embeddings["has_bio"]
    img_miss_val = val_embeddings["image_missing"]

    X_img_test = test_embeddings["image_embs"]
    X_txt_test = test_embeddings["text_embs"]
    y_test = test_embeddings["labels"]
    has_test = test_embeddings["has_bio"]
    img_miss_test = test_embeddings["image_missing"]

    def make_input(img, txt, has_bio, image_missing):
        """Concatenate [img_emb, txt_emb, has_bio, image_missing]"""
        return np.concatenate([img, txt, has_bio.reshape(-1, 1), image_missing.reshape(-1, 1)], axis=1).astype(np.float32)

    X_tr = make_input(X_img_tr, X_txt_tr, has_tr, img_miss_tr)
    X_val = make_input(X_img_val, X_txt_val, has_val, img_miss_val)
    X_test = make_input(X_img_test, X_txt_test, has_test, img_miss_test)

    image_dim = X_img_tr.shape[1]
    text_dim = X_txt_tr.shape[1]

    input_dim = X_tr.shape[1]
    model = MLPClassifierWithMissing(image_emb_dim=image_dim, text_emb_dim=text_dim, hidden_dims=(1024, 256), dropout=0.3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    # Forming the Dataset for the MLP
    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32)))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val.astype(np.float32)))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test.astype(np.float32)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    best_val_auc = -1.0
    best_state = None

    # Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            # modality dropout (optional): drop text embedding randomly
            if modality_dropout > 0:
                # text slice
                txt_start = image_dim
                txt_end = image_dim + text_dim
                mask = (torch.rand(xb.size(0), device=xb.device) < modality_dropout).float().unsqueeze(1)
                xb[:, txt_start:txt_end] = xb[:, txt_start:txt_end] * (1.0 - mask)

            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)

        avg_loss = total_loss / n

        # Evaluation Loop
        model.eval()
        val_logits_list = []
        val_labels_list = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                val_logits_list.append(logits.cpu().numpy())
                val_labels_list.append(yb.numpy())

        val_logits = np.concatenate(val_logits_list)
        val_labels = np.concatenate(val_labels_list)
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))

        try:
            val_auc = roc_auc_score(val_labels, val_probs)
            val_pr = average_precision_score(val_labels, val_probs)
        except Exception:
            val_auc = float("nan")
            val_pr = float("nan")

        print(f"Epoch {epoch}/{epochs} train_loss={avg_loss:.4f} val_auc={val_auc:.4f} val_pr={val_pr:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "epoch": epoch,
                "val_auc": val_auc,
            }

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "mlp_model.pth")
    torch.save(best_state, model_path)
    print(f"[+] Saved best model to {model_path} (val_auc={best_val_auc:.4f})")

    # Evaluate on test set with the best_state loaded
    model.load_state_dict(best_state["model_state"])
    model.eval()
    logits_list = []
    y_list = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            logits_list.append(logits.cpu().numpy())
            y_list.append(yb.numpy())
    logits = np.concatenate(logits_list)
    y_true = np.concatenate(y_list)
    y_prob = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (y_prob >= 0.5).astype(int)

    overall_metrics = compute_metrics(y_true, y_prob, y_pred)
    print("\n[TEST - overall]")
    print_metrics(overall_metrics)

    # Subset metrics: has_bio vs no_bio
    print("\n[TEST - by bio presence]")
    idx_has = (X_test[:, -2] == 1)  # has_bio now at -2
    idx_no = (X_test[:, -2] == 0)
    if idx_has.sum() > 0:
        m_has = compute_metrics(y_true[idx_has], y_prob[idx_has], y_pred[idx_has])
        print("Has bio:")
        print_metrics(m_has)
    if idx_no.sum() > 0:
        m_no = compute_metrics(y_true[idx_no], y_prob[idx_no], y_pred[idx_no])
        print("No bio:")
        print_metrics(m_no)

    # Subset metrics: image present vs missing
    print("\n[TEST - by image presence]")
    idx_img_present = (X_test[:, -1] == 0)
    idx_img_missing = (X_test[:, -1] == 1)
    if idx_img_present.sum() > 0:
        m_ip = compute_metrics(y_true[idx_img_present], y_prob[idx_img_present], y_pred[idx_img_present])
        print("Image present:")
        print_metrics(m_ip)
    if idx_img_missing.sum() > 0:
        m_im = compute_metrics(y_true[idx_img_missing], y_prob[idx_img_missing], y_pred[idx_img_missing])
        print("Image missing:")
        print_metrics(m_im)

    npz_path = os.path.join(out_dir, "test_preds.npz")
    np.savez(npz_path, y_true=y_true, y_prob=y_prob, y_pred=y_pred)
    print(f"[+] Saved test predictions to {npz_path}")

    return model_path

# ------------------------------
# Metrics helpers
# ------------------------------
def compute_metrics(y_true, y_prob, y_pred):
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    try:
        pr = average_precision_score(y_true, y_prob)
    except Exception:
        pr = float("nan")
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"auc": auc, "pr": pr, "precision": precision, "recall": recall, "f1": f1, "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}

def print_metrics(metrics: Dict[str, Any]):
    print(f" AUC:       {metrics['auc']:.4f}")
    print(f" PR-AUC:    {metrics['pr']:.4f}")
    print(f" Precision: {metrics['precision']:.4f}")
    print(f" Recall:    {metrics['recall']:.4f}")
    print(f" F1:        {metrics['f1']:.4f}")
    print(f" TP/FP/TN/FN: {metrics['tp']}/{metrics['fp']}/{metrics['tn']}/{metrics['fn']}")

def compute_image_missing_contingency(
    df: pd.DataFrame,
    image_col: str,
    label_col: str,
    image_cache: str = "image_cache",
    save_csv: str = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compute contingency table of image_missing (0/1) vs label, print results,
    run Chi-square test (if scipy installed), and optionally save results to CSV.

    Returns a dict with keys:
      - 'table' : pandas DataFrame (contingency counts)
      - 'prop_table' : contingency normalized by row (proportions of labels per missingness)
      - 'chi2_result' : dict with chi2,p,dof,expected (if scipy available) else None
    """
    # determine missingness: for URL entries check cache presence; for local paths check file existence
    def _is_missing(ref):
        try:
            if isinstance(ref, str) and ref.lower().startswith(("http://", "https://")):
                cache_path = url_to_cache_path(ref, image_cache)
                return 0 if os.path.exists(cache_path) else 1
            else:
                return 0 if os.path.exists(str(ref)) else 1
        except Exception:
            return 1

    # Apply quickly (vectorized-ish)
    refs = df[image_col].fillna("").astype(str)
    # compute boolean missing (0 present, 1 missing)
    image_missing = refs.apply(_is_missing).astype(int)
    tmp = df.copy()
    tmp = tmp.assign(image_missing=image_missing)

    # contingency table (counts)
    table = pd.crosstab(tmp["image_missing"], tmp[label_col])
    # proportions by missingness row
    prop_table = pd.crosstab(tmp["image_missing"], tmp[label_col], normalize="index")

    if verbose:
        print("\n[Image missing vs Label contingency]")
        print(table)
        print("\n[Proportions by image_missing (rows sum to 1)]")
        print(prop_table)
        # print basic marginals
        totals = table.sum(axis=1)
        print("\nCounts by image_missing:")
        for miss_val in totals.index:
            print(f"  image_missing={miss_val}: {int(totals.loc[miss_val])} samples")

    chi2_result = None
    # run chi-square if scipy is available and table is at least 2x2
    try:
        if table.shape[0] >= 2 and table.shape[1] >= 2:
            chi2, p, dof, expected = chi2_contingency(table.values)
            chi2_result = {"chi2": float(chi2), "p_value": float(p), "dof": int(dof), "expected": expected}
            if verbose:
                print(f"\n[Chi-square test] chi2={chi2:.4f}, p={p:.4e}, dof={dof}")
                # quick interpretation
                alpha = 0.05
                if p < alpha:
                    print(f"  -> p < {alpha}: reject independence (missingness and label are likely associated)")
                else:
                    print(f"  -> p >= {alpha}: cannot reject independence (no strong evidence of association)")
        else:
            print("\n[Note] contingency table too small for chi-square (needs at least 2x2).")
    except Exception as e:
        print(f"\n[!] Chi-square test failed: {e}")

    # save csv if requested
    if save_csv:
        try:
            os.makedirs(os.path.dirname(save_csv), exist_ok=True)
            # save counts and proportions side by side
            out_df = table.copy()
            # append proportions columns with suffix
            prop_df = prop_table.copy()
            prop_df.columns = [f"{c}_prop" for c in prop_df.columns]
            # join (note: index is image_missing)
            out_df = out_df.join(prop_df, how="left")
            out_df.to_csv(save_csv, index=True)
            if verbose:
                print(f"\n[+] Saved contingency table to {save_csv}")
        except Exception as e:
            print(f"[!] Failed to save contingency CSV: {e}")

    return {"table": table, "prop_table": prop_table, "chi2_result": chi2_result, "tmp": tmp}

# ------------------------------
# Main entry (wires everything together)
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train multimodal Twitter-bot detector supporting async image prefetch")
    parser.add_argument("--csv", required=True, help="Input CSV with image URLs/paths, bio, label")
    parser.add_argument("--image-col", default="profile_image_url", help="CSV column for image path/URL")
    parser.add_argument("--bio-col", default="description", help="CSV column for bio")
    parser.add_argument("--label-col", default="account_type", help="CSV column for label (0/1)")
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-embeddings", action="store_true", help="Save embeddings to disk (out_dir/embeddings.pt)")
    parser.add_argument("--modality-dropout", type=float, default=0.2, help="Probability to drop text modality during training")
    parser.add_argument("--image-cache", default="image_cache", help="Directory to cache downloaded images")
    parser.add_argument("--prefetch-images", action="store_true", help="Prefetch all unique images into cache before embedding extraction")
    parser.add_argument("--prefetch-concurrency", type=int, default=32, help="Max concurrent HTTP requests for prefetch")
    parser.add_argument("--prefetch-timeout", type=float, default=6.0, help="Per-request timeout in seconds for prefetch")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.image_cache, exist_ok=True)

    print("[*] Loading CSV...")
    df = pd.read_csv(args.csv)
    if args.label_col not in df.columns:
        raise ValueError(f"Label column {args.label_col} not found in CSV")
    if args.image_col not in df.columns:
        raise ValueError(f"Image column {args.image_col} not found in CSV")
    if args.bio_col not in df.columns:
        raise ValueError(f"Bio column {args.bio_col} not found in CSV")

    df[args.bio_col] = df[args.bio_col].fillna("").astype(str)
    # map labels if needed (bot/human strings)
    if df[args.label_col].dtype == object or df[args.label_col].dtype == "O":
        df[args.label_col] = df[args.label_col].map({'bot':1, 'human':0})

    # Optional async prefetch
    if args.prefetch_images:
        print("[*] Starting async prefetch. This may take time depending on number of URLs.")
        asyncio.run(async_prefetch_image_cache(df, args.image_col, args.image_cache,
                                              max_workers=args.prefetch_concurrency,
                                              max_bytes=5_000_000,
                                              timeout=args.prefetch_timeout))
    else:
        print("[*] Skipping prefetch. Images will be fetched on-the-fly (slower).")

    contingency_out = compute_image_missing_contingency(
        df,
        image_col=args.image_col,
        label_col=args.label_col,
        image_cache=args.image_cache,
        save_csv=os.path.join(args.out_dir, "image_missing_contingency.csv"),
        verbose=True,
       )
    
    trainval_df, test_df = train_test_split(df, test_size=0.10, random_state=args.seed, stratify=df[args.label_col])
    train_df, val_df = train_test_split(trainval_df, test_size=0.20, random_state=args.seed, stratify=trainval_df[args.label_col])
    print(f"Dataset sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    device = args.device

    print("[*] Loading CLIP (image) model and processor...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print("[*] Loading text encoder (DistilRoBERTa)...")
    text_model_name = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    text_model = AutoModel.from_pretrained(text_model_name).to(device)

    emb_path = os.path.join(args.out_dir, "embeddings.pt")
    if os.path.exists(emb_path) and args.save_embeddings:
        print(f"[+] Loading saved embeddings from {emb_path}")
        all_emb = torch.load(emb_path, weights_only=False)
        train_emb = all_emb["train"]
        val_emb = all_emb["val"]
        test_emb = all_emb["test"]
    else:
        print("[*] Extracting train embeddings...")
        train_out = extract_embeddings(train_df, args.image_col, args.bio_col, args.label_col,
                                       clip_model, clip_processor, text_model, tokenizer, device,
                                       batch_size=args.batch_size, max_length=64, image_cache=args.image_cache, save_path=None)
        print("[*] Extracting val embeddings...")
        val_out = extract_embeddings(val_df, args.image_col, args.bio_col, args.label_col,
                                     clip_model, clip_processor, text_model, tokenizer, device,
                                     batch_size=args.batch_size, max_length=64, image_cache=args.image_cache, save_path=None)
        print("[*] Extracting test embeddings...")
        test_out = extract_embeddings(test_df, args.image_col, args.bio_col, args.label_col,
                                      clip_model, clip_processor, text_model, tokenizer, device,
                                      batch_size=args.batch_size, max_length=64, image_cache=args.image_cache, save_path=None)

        train_emb = train_out
        val_emb = val_out
        test_emb = test_out

        if args.save_embeddings:
            torch.save({"train": train_emb, "val": val_emb, "test": test_emb}, emb_path)
            print(f"[+] Saved all embeddings to {emb_path}")

    model_path = train_classifier(
        train_embeddings=train_emb,
        val_embeddings=val_emb,
        test_embeddings=test_emb,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=1e-4,
        modality_dropout=args.modality_dropout,
        device=device,
    )
    print("[+] Done. model saved at:", model_path)

if __name__ == "__main__":
    main()
