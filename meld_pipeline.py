"""
MELD multimodal pipeline utilities.

  - Run feature extraction, unimodal training, multimodal training, and
    evaluation for MELD dataset.
  - Override dataset/feature locations via environment variables:
        MELD_DATASET_ROOT (default: ".")
        MELD_FEATURES_ROOT (default: "./features_meld")
"""

import os
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from glob import glob
from typing import List, Dict, Tuple, Optional, Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchaudio
import soundfile as sf
import cv2

from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel, AutoProcessor


# ---------------------------------------------------------------------
# Runtime setup
# ---------------------------------------------------------------------
def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


device = _get_device()
print("Torch:", torch.__version__)
print("Device:", device)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class Paths:
    # project root that contains MELD-RAW/
    dataset_root: Path = Path(os.getenv("MELD_DATASET_ROOT", "../MELD-RAW/MELD.Raw")).resolve()

    # Where to store extracted feature .pt files
    features_root: Path = Path(os.getenv("MELD_FEATURES_ROOT", "../MELD-Features-Models-498K/MELD.Features.Models")).resolve()


@dataclass
class ModelNames:
    text_model: str = "roberta-base"
    audio_model: str = "facebook/wav2vec2-base-960h"
    video_model: str = "google/vit-base-patch16-224-in21k"


@dataclass
class TrainConfig:
    num_classes: int = 7
    max_text_len: int = 64
    wav_target_sr: int = 16000
    num_video_frames: int = 5
    batch_size: int = 4
    lr: float = 2e-5
    num_epochs: int = 3  # keep small for runtime limits
    device: str = str(device)


paths = Paths()
model_names = ModelNames()
train_cfg = TrainConfig()


# ---------------------------------------------------------------------
# Label map (MELD emotions)
# ---------------------------------------------------------------------
EMOTIONS = [
    "neutral",
    "joy",
    "sadness",
    "anger",
    "surprise",
    "fear",
    "disgust",
]
label2id = {lbl: i for i, lbl in enumerate(EMOTIONS)}
id2label = {i: lbl for lbl, i in label2id.items()}
print("Label map:", label2id)


# ---------------------------------------------------------------------
# CSV locator – supports both MELD-RAW and /data/MELD layout
# ---------------------------------------------------------------------
def find_meld_csv(dataset_root: Path, split: str) -> Path:
    """
    Locate MELD CSV files for a given split.
    split in {'train', 'dev', 'test'}.
    """
    split = split.lower()
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"Unknown split: {split}")

    # 1) Kaggle MELD-RAW layout
    path1 = dataset_root / "MELD-RAW" / "MELD.Raw" / split / f"{split}_sent_emo.csv"
    if path1.exists():
        print(f"[find_meld_csv] Using (MELD-RAW layout): {path1}")
        return path1

    # 2) GitHub MELD layout
    path2 = dataset_root / "data" / "MELD" / f"{split}_sent_emo.csv"
    if path2.exists():
        print(f"[find_meld_csv] Using (data/MELD layout): {path2}")
        return path2

    # 3) Fallback: scan for *{split}_sent_emo*.csv anywhere
    pattern = f"*{split}_sent_emo*.csv"
    matches = list(dataset_root.rglob(pattern))
    if matches:
        print(f"[find_meld_csv] Using (rglob): {matches[0]}")
        return matches[0]

    # Debug failure
    print(f"[find_meld_csv] Could not find expected paths:")
    print("  -", path1)
    print("  -", path2)
    all_csvs = list(dataset_root.rglob("*.csv"))
    if not all_csvs:
        print("[find_meld_csv] No CSV files found under", dataset_root)
    else:
        print("[find_meld_csv] Available CSV files (first 20):")
        for p in all_csvs[:20]:
            print("  -", p)
    raise FileNotFoundError(f"No CSV found for split={split} (pattern {pattern})")


# ---------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------
class TextEncoder(nn.Module):
    def __init__(self, model_name: str = model_names.text_model, max_len: int = train_cfg.max_text_len):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.max_len = max_len

    @torch.inference_mode()
    def encode(self, text: str) -> torch.Tensor:
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        ).to(device)
        out = self.model(**enc)
        cls_emb = out.last_hidden_state[:, 0, :]  # (1, hidden)
        return cls_emb.squeeze(0).cpu()


class AudioEncoder(nn.Module):
    def __init__(self, model_name: str = model_names.audio_model, target_sr: int = train_cfg.wav_target_sr):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.target_sr = target_sr

    def _load_wav(self, path: str) -> torch.Tensor:
        """
        Load audio with torchaudio; fall back to soundfile if torchaudio backend
        is unavailable. Always returns a 1-D mono tensor at target_sr.
        """
        try:
            wav, sr = torchaudio.load(path)  # (C, T)
        except Exception:
            # Fallback for macOS builds where torchaudio backend is missing.
            wav_np, sr = sf.read(path, dtype="float32", always_2d=True)  # (T, C)
            wav = torch.from_numpy(wav_np).T  # (C, T)

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        return wav.squeeze(0)

    @torch.inference_mode()
    def encode(self, wav_path: str) -> torch.Tensor:
        wav = self._load_wav(wav_path)
        inputs = self.processor(
            wav,
            sampling_rate=self.target_sr,
            return_tensors="pt",
        ).to(device)
        out = self.model(**inputs)
        emb = out.last_hidden_state.mean(dim=1)  # (1, hidden)
        return emb.squeeze(0).cpu()


class VideoEncoder(nn.Module):
    def __init__(self, model_name: str = model_names.video_model, num_frames: int = train_cfg.num_video_frames):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.num_frames = num_frames

    def _sample_frames(self, video_path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video {video_path}")

        frames = []
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if length <= 0:
            cap.release()
            raise RuntimeError(f"Video {video_path} has no frames")

        idxs = np.linspace(0, length - 1, num=self.num_frames, dtype=int)
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"No frames read from video {video_path}")
        return frames

    @torch.inference_mode()
    def encode(self, video_path: str) -> torch.Tensor:
        frames = self._sample_frames(video_path)
        inputs = self.processor(
            images=frames,
            return_tensors="pt",
        ).to(device)
        out = self.model(**inputs)
        cls_tokens = out.last_hidden_state[:, 0, :]  # (F, hidden)
        emb = cls_tokens.mean(dim=0)
        return emb.cpu()


# ---------------------------------------------------------------------
# Audio / video file indexing
# ---------------------------------------------------------------------
def build_av_indices(dataset_root: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    print("Indexing audio and video files under:", dataset_root)
    wav_paths = list(dataset_root.rglob("*.wav"))
    mp4_paths = list(dataset_root.rglob("*.mp4"))
    print(f"  Found {len(wav_paths)} wav files, {len(mp4_paths)} mp4 files")

    wav_index = {p.name: p for p in wav_paths}
    mp4_index = {p.name: p for p in mp4_paths}
    return wav_index, mp4_index


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------
def extract_split(
    df: pd.DataFrame,
    split: str,
    text_enc: TextEncoder,
    audio_enc: AudioEncoder,
    video_enc: VideoEncoder,
    wav_index: Dict[str, Path],
    mp4_index: Dict[str, Path],
    features_root: Path = paths.features_root,
):
    out_dir = features_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extract {split}"):
        text = row["Utterance"]
        emo = str(row["Emotion"]).lower()
        if emo not in label2id:
            continue
        label = label2id[emo]

        dia_id = int(row["Dialogue_ID"])
        utt_id = int(row["Utterance_ID"])
        speaker = str(row.get("Speaker", ""))

        base_name = f"dia{dia_id}_utt{utt_id}"
        audio_name = base_name + ".wav"
        video_name = base_name + ".mp4"

        audio_path = wav_index.get(audio_name, None)
        video_path = mp4_index.get(video_name, None)

        ht = text_enc.encode(text)

        if audio_path is not None and audio_path.exists():
            try:
                ha = audio_enc.encode(str(audio_path))
            except Exception as e:
                print(f"Error encoding audio {audio_path}: {e}")
                ha = torch.zeros_like(ht)
        else:
            print(f"No audio path found for {audio_name}")
            ha = torch.zeros_like(ht)

        if video_path is not None and video_path.exists():
            try:
                hv = video_enc.encode(str(video_path))
            except Exception:
                print(f"Error encoding video {video_path}")
                hv = torch.zeros_like(ht)
        else:
            print(f"No video path found for {video_name}")
            hv = torch.zeros_like(ht)

        out = {
            "text_emb": ht,
            "audio_emb": ha,
            "video_emb": hv,
            "label": label,
            "dialogue_id": dia_id,
            "utterance_id": utt_id,
            "speaker": speaker,
        }
        fname = out_dir / f"{base_name}.pt"
        torch.save(out, fname)


def run_feature_extraction(
    paths: Paths = paths,
    model_names: ModelNames = model_names,
    train_cfg: TrainConfig = train_cfg,
):
    """Extract text/audio/video embeddings and save to disk."""
    paths.features_root.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(find_meld_csv(paths.dataset_root, "train"))
    dev_df = pd.read_csv(find_meld_csv(paths.dataset_root, "dev"))
    test_df = pd.read_csv(find_meld_csv(paths.dataset_root, "test"))
    print("Loaded CSVs:", len(train_df), len(dev_df), len(test_df))

    wav_index, mp4_index = build_av_indices(paths.dataset_root)

    print("Loading encoders...")
    text_enc = TextEncoder(model_name=model_names.text_model, max_len=train_cfg.max_text_len)
    audio_enc = AudioEncoder(model_name=model_names.audio_model, target_sr=train_cfg.wav_target_sr)
    video_enc = VideoEncoder(model_name=model_names.video_model, num_frames=train_cfg.num_video_frames)
    print("Encoders ready.")

    extract_split(train_df, "train", text_enc, audio_enc, video_enc, wav_index, mp4_index, paths.features_root)
    extract_split(dev_df, "dev", text_enc, audio_enc, video_enc, wav_index, mp4_index, paths.features_root)
    extract_split(test_df, "test", text_enc, audio_enc, video_enc, wav_index, mp4_index, paths.features_root)
    print("Feature extraction completed.")


# ---------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------
class UtteranceDataset(Dataset):
    """Single-utterance dataset for unimodal models."""

    def __init__(self, split: str, modality: str = "text", features_root: Path = paths.features_root):
        assert modality in {"text", "audio", "video"}
        self.modality = modality
        self.files = sorted(glob(str(features_root / split / "*.pt")))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = torch.load(self.files[idx])
        if self.modality == "text":
            x = data["text_emb"]
        elif self.modality == "audio":
            x = data["audio_emb"]
        else:
            x = data["video_emb"]

        return {
            "x": x.float(),
            "label": torch.tensor(data["label"], dtype=torch.long),
        }


class ConversationDataset(Dataset):
    """Dialogue-level dataset for contextual models."""

    def __init__(self, split: str, features_root: Path = paths.features_root):
        files = glob(str(features_root / split / "*.pt"))
        dialogues: Dict[int, List[Tuple[int, Dict]]] = {}
        for f in files:
            d = torch.load(f)
            d_id = int(d["dialogue_id"])
            u_id = int(d["utterance_id"])
            if d_id not in dialogues:
                dialogues[d_id] = []
            dialogues[d_id].append((u_id, d))

        self.dialogues: List[Tuple[int, List[Dict]]] = []
        for d_id, lst in dialogues.items():
            lst_sorted = [x[1] for x in sorted(lst, key=lambda t: t[0])]
            self.dialogues.append((d_id, lst_sorted))
        self.dialogues.sort(key=lambda t: t[0])

    def __len__(self) -> int:
        return len(self.dialogues)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        d_id, utterances = self.dialogues[idx]
        text_embs = torch.stack([u["text_emb"].float() for u in utterances])
        audio_embs = torch.stack([u["audio_emb"].float() for u in utterances])
        video_embs = torch.stack([u["video_emb"].float() for u in utterances])
        labels = torch.tensor([u["label"] for u in utterances], dtype=torch.long)
        return {
            "dialogue_id": d_id,
            "text_embs": text_embs,
            "audio_embs": audio_embs,
            "video_embs": video_embs,
            "labels": labels,
        }


def collate_conversations(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    max_len = max(len(b["labels"]) for b in batch)

    def pad_sequence(seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        D = seqs[0].shape[-1]
        B = len(seqs)
        out = torch.zeros(B, max_len, D, dtype=seqs[0].dtype)
        attn_mask = torch.zeros(B, max_len, dtype=torch.bool)
        for i, s in enumerate(seqs):
            L = s.shape[0]
            out[i, :L] = s
            attn_mask[i, :L] = 1
        return out, attn_mask

    text_embs, txt_mask = pad_sequence([b["text_embs"] for b in batch])
    audio_embs, _ = pad_sequence([b["audio_embs"] for b in batch])
    video_embs, _ = pad_sequence([b["video_embs"] for b in batch])

    B = len(batch)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["labels"].shape[0]
        labels[i, :L] = b["labels"]

    return {
        "text_embs": text_embs,
        "audio_embs": audio_embs,
        "video_embs": video_embs,
        "labels": labels,
        "attn_mask": txt_mask,
    }


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
class UnimodalClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        num_classes: int = train_cfg.num_classes,
        hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EarlyFusionContext(nn.Module):
    """Early fusion of T/A/V embeddings + Transformer encoder."""

    def __init__(
        self,
        input_dim_each: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = train_cfg.num_classes,
        dropout: float = 0.1,
    ):
        super().__init__()
        fused_dim = input_dim_each * 3
        self.proj = nn.Linear(fused_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        text_embs: torch.Tensor,
        audio_embs: torch.Tensor,
        video_embs: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat([text_embs, audio_embs, video_embs], dim=-1)
        x = self.proj(fused)
        src_key_padding_mask = ~attn_mask
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.classifier(h)
        return logits


class LateFusionContext(nn.Module):
    """Per-modality Transformer branches + probability-level fusion."""

    def __init__(
        self,
        input_dim_each: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = train_cfg.num_classes,
        w_text: float = 0.5,
        w_audio: float = 0.25,
        w_video: float = 0.25,
        dropout: float = 0.1,
        learnable_weights: bool = True,
    ):
        super().__init__()
        self.text_proj = nn.Linear(input_dim_each, hidden_dim)
        self.audio_proj = nn.Linear(input_dim_each, hidden_dim)
        self.video_proj = nn.Linear(input_dim_each, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder_t = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder_a = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder_v = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.clf_t = nn.Linear(hidden_dim, num_classes)
        self.clf_a = nn.Linear(hidden_dim, num_classes)
        self.clf_v = nn.Linear(hidden_dim, num_classes)

        self.learnable_weights = learnable_weights
        if learnable_weights:
            # Initialize logits from provided priors so softmax ≈ given weights.
            init = torch.tensor([w_text, w_audio, w_video], dtype=torch.float)
            init = torch.log(init / init.sum() + 1e-8)
            self.weight_logits = nn.Parameter(init)
        else:
            self.register_buffer("fixed_weights", torch.tensor([w_text, w_audio, w_video], dtype=torch.float))

    def forward(
        self,
        text_embs: torch.Tensor,
        audio_embs: torch.Tensor,
        video_embs: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        src_key_padding_mask = ~attn_mask

        t = self.text_proj(text_embs)
        t = self.encoder_t(t, src_key_padding_mask=src_key_padding_mask)
        logits_t = self.clf_t(t)

        a = self.audio_proj(audio_embs)
        a = self.encoder_a(a, src_key_padding_mask=src_key_padding_mask)
        logits_a = self.clf_a(a)

        v = self.video_proj(video_embs)
        v = self.encoder_v(v, src_key_padding_mask=src_key_padding_mask)
        logits_v = self.clf_v(v)

        probs_t = logits_t.softmax(dim=-1)
        probs_a = logits_a.softmax(dim=-1)
        probs_v = logits_v.softmax(dim=-1)

        if self.learnable_weights:
            weights = torch.softmax(self.weight_logits, dim=0)  # (3,)
        else:
            weights = self.fixed_weights

        probs = weights[0] * probs_t + weights[1] * probs_a + weights[2] * probs_v
        logits = probs.clamp(min=1e-8).log()
        return logits


# ---------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------
def train_unimodal(
    modality: str,
    *,
    hidden: int = 256,
    dropout: float = 0.2,
    lr: float = train_cfg.lr,
    weight_decay: float = 0.0,
    num_epochs: int = train_cfg.num_epochs,
    batch_size: int = train_cfg.batch_size,
    use_class_weights: bool = True,
    optimizer_cls: Callable = torch.optim.AdamW,
    optimizer_kwargs: Optional[Dict] = None,
):
    """Train a unimodal classifier on pre-extracted embeddings."""
    print(f"\n=== Unimodal training: {modality} ===")
    train_ds = UtteranceDataset("train", modality=modality, features_root=paths.features_root)
    dev_ds = UtteranceDataset("dev", modality=modality, features_root=paths.features_root)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size)

    model = UnimodalClassifier(hidden=hidden, dropout=dropout).to(device)
    opt_kwargs = optimizer_kwargs or {"lr": lr, "weight_decay": weight_decay}
    optimizer = optimizer_cls(model.parameters(), **opt_kwargs)

    # Optional class weighting to mitigate imbalance-driven collapse.
    if use_class_weights:
        counts = Counter(torch.load(f)["label"] for f in train_ds.files)
        total = sum(counts.values())
        num_classes = train_cfg.num_classes
        weights = torch.tensor(
            [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)],
            dtype=torch.float,
            device=device,
        )
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("Using class weights:", weights.cpu().numpy())
    else:
        criterion = nn.CrossEntropyLoss()

    best_f1 = -float("inf")
    best_path = f"unimodal_{modality}_best.pt"

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        train_correct = 0
        train_count = 0
        for batch in tqdm(train_loader, desc=f"{modality} Epoch {epoch+1}"):
            x = batch["x"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == y).sum().item()
            train_count += x.size(0)
        avg_loss = total_loss / len(train_ds)
        train_acc = train_correct / max(train_count, 1)
        print(f"[{modality}] Epoch {epoch+1} Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        dev_total_loss = 0.0
        dev_total = 0
        with torch.no_grad():
            for batch in dev_loader:
                x = batch["x"].to(device)
                y = batch["label"]
                logits = model(x)
                loss = criterion(logits, y.to(device))
                dev_total_loss += loss.item() * x.size(0)
                dev_total += x.size(0)
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(y.cpu().numpy().tolist())
        f1 = f1_score(all_labels, all_preds, average="weighted")
        acc = (np.array(all_preds) == np.array(all_labels)).mean()
        dev_loss = dev_total_loss / max(dev_total, 1)
        print(f"[{modality}] Epoch {epoch+1} Dev Loss: {dev_loss:.4f} | Dev F1: {f1:.4f} | Dev Acc: {acc:.4f}")

        # Track and save best-by-F1 checkpoint across epochs.
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_path)
            print(f"[{modality}] New best Dev F1={best_f1:.4f} → saved {best_path}")

    print(f"Finished {modality} training. Best Dev F1={best_f1:.4f} (saved at {best_path})")


def run_epoch_multimodal(model, loader, optimizer=None):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    if optimizer is not None:
        model.train()
        total_loss = 0.0
        total_count = 0
        correct = 0
        for batch in tqdm(loader, desc="Train"):
            text = batch["text_embs"].to(device)
            audio = batch["audio_embs"].to(device)
            video = batch["video_embs"].to(device)
            labels = batch["labels"].to(device)
            attn_mask = batch["attn_mask"].to(device)

            optimizer.zero_grad()
            logits = model(text, audio, video, attn_mask)
            B, L, C = logits.shape
            loss = criterion(logits.view(B * L, C), labels.view(-1))
            loss.backward()
            optimizer.step()

            n_valid = (labels != -100).sum().item()
            total_loss += loss.item() * n_valid
            total_count += n_valid
            preds = logits.argmax(dim=-1)
            correct += ((preds == labels) & (labels != -100)).sum().item()
        acc = correct / max(total_count, 1)
        return total_loss / max(total_count, 1), acc
    else:
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Eval"):
                text = batch["text_embs"].to(device)
                audio = batch["audio_embs"].to(device)
                video = batch["video_embs"].to(device)
                labels = batch["labels"]  # cpu
                attn_mask = batch["attn_mask"].to(device)

                logits = model(text, audio, video, attn_mask)
                preds = logits.argmax(dim=-1).cpu()

                valid = labels != -100
                all_labels.extend(labels[valid].numpy().tolist())
                all_preds.extend(preds[valid].numpy().tolist())
        f1 = f1_score(all_labels, all_preds, average="weighted")
        acc = (np.array(all_preds) == np.array(all_labels)).mean()
        return f1, acc


def train_multimodal(
    model_type: str = "early",
    *,
    hidden_dim: int = 256,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    lr: float = train_cfg.lr,
    weight_decay: float = 0.0,
    num_epochs: int = train_cfg.num_epochs,
    batch_size: int = train_cfg.batch_size,
    w_text: float = 0.5,
    w_audio: float = 0.25,
    w_video: float = 0.25,
    learnable_weights: bool = True,
    optimizer_cls: Callable = torch.optim.AdamW,
    optimizer_kwargs: Optional[Dict] = None,
):
    """Train early- or late-fusion contextual models."""
    print(f"\n=== Multimodal training ({model_type} fusion) ===")
    train_ds = ConversationDataset("train", features_root=paths.features_root)
    dev_ds = ConversationDataset("dev", features_root=paths.features_root)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_conversations,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_conversations,
    )

    if model_type == "early":
        model = EarlyFusionContext(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)
    elif model_type == "late":
        model = LateFusionContext(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            w_text=w_text,
            w_audio=w_audio,
            w_video=w_video,
            learnable_weights=learnable_weights,
        ).to(device)
    else:
        raise ValueError("model_type must be 'early' or 'late'")

    opt_kwargs = optimizer_kwargs or {"lr": lr, "weight_decay": weight_decay}
    optimizer = optimizer_cls(model.parameters(), **opt_kwargs)

    best_f1 = -float("inf")
    best_path = f"multimodal_{model_type}_fusion_best.pt"

    for epoch in range(num_epochs):
        train_loss, train_acc = run_epoch_multimodal(model, train_loader, optimizer)
        dev_f1, dev_acc = run_epoch_multimodal(model, dev_loader, optimizer=None)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Train Acc={train_acc:.4f} | Dev F1={dev_f1:.4f} | Dev Acc={dev_acc:.4f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            torch.save(model.state_dict(), best_path)
            print(f"[{model_type}] New best Dev F1={best_f1:.4f} → saved {best_path}")

    print(f"Finished multimodal {model_type} training. Best Dev F1={best_f1:.4f} (saved at {best_path})")


# ---------------------------------------------------------------------
# Evaluation + plotting
# ---------------------------------------------------------------------
def eval_unimodal_model(
    modality: str,
    model_path: str,
    split: str = "dev",
    *,
    hidden: int = 512,
    dropout: float = 0.2,
):
    """Evaluate a saved unimodal model."""
    ds = UtteranceDataset(split, modality=modality, features_root=paths.features_root)
    dl = DataLoader(ds, batch_size=train_cfg.batch_size)

    model = UnimodalClassifier(hidden=hidden, dropout=dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in dl:
            x = batch["x"].to(device)
            y = batch["label"].cpu().numpy()
            logits = model(x)
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_labels.extend(y.tolist())
            all_preds.extend(preds.tolist())

    f1 = f1_score(all_labels, all_preds, average="weighted")
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    return f1, acc, all_labels, all_preds


def eval_multimodal_model(
    model_type: str,
    model_path: str,
    split: str = "dev",
    *,
    hidden_dim: int = 512,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.2,
    w_text: float = 0.5,
    w_audio: float = 0.25,
    w_video: float = 0.25,
):
    """Evaluate a saved multimodal model."""
    ds = ConversationDataset(split, features_root=paths.features_root)
    dl = DataLoader(
        ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        collate_fn=collate_conversations,
    )

    if model_type == "early":
        model = EarlyFusionContext(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)
    elif model_type == "late":
        model = LateFusionContext(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            w_text=w_text,
            w_audio=w_audio,
            w_video=w_video,
        ).to(device)
    else:
        raise ValueError("model_type must be 'early' or 'late'")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in dl:
            text = batch["text_embs"].to(device)
            audio = batch["audio_embs"].to(device)
            video = batch["video_embs"].to(device)
            labels = batch["labels"]  # cpu
            attn_mask = batch["attn_mask"].to(device)

            logits = model(text, audio, video, attn_mask)
            preds = logits.argmax(dim=-1).cpu()

            valid = labels != -100
            all_labels.extend(labels[valid].numpy().tolist())
            all_preds.extend(preds[valid].numpy().tolist())

    f1 = f1_score(all_labels, all_preds, average="weighted")
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    return f1, acc, all_labels, all_preds


def plot_f1_bar(f1_dict, title="Dev weighted F1 scores"):
    names = list(f1_dict.keys())
    values = [f1_dict[k] for k in names]

    plt.figure(figsize=(8, 4))
    plt.bar(names, values)
    plt.ylabel("Weighted F1")
    plt.title(title)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_confusion(labels, preds, title="Confusion matrix (dev set)"):
    cm = confusion_matrix(labels, preds, labels=list(range(len(EMOTIONS))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMOTIONS)
    plt.figure(figsize=(6, 6))
    disp.plot(include_values=True, cmap=None, xticks_rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Convenience runner (optional)
# ---------------------------------------------------------------------
def run_pipeline(
    do_extract_features: bool = False,
    do_train_unimodal: bool = False,
    do_train_mm_early: bool = False,
    do_train_mm_late: bool = False,
    do_plot_eval: bool = False,
):
    """
    Mirror the original draft.py flags but callable from scripts.
    """
    if do_extract_features:
        run_feature_extraction()

    if do_train_unimodal:
        for modality in ["text", "audio", "video"]:
            train_unimodal(modality)

    if do_train_mm_early:
        train_multimodal("early")

    if do_train_mm_late:
        train_multimodal("late")

    if do_plot_eval:
        results = {}
        # Unimodal
        try:
            f1_t, acc_t, lab_t, pred_t = eval_unimodal_model("text", "unimodal_text.pt")
            results["uni_text"] = f1_t
            print("Unimodal text F1:", f1_t, "Acc:", acc_t)
        except FileNotFoundError as e:
            print("Text model missing:", e)

        try:
            f1_a, acc_a, lab_a, pred_a = eval_unimodal_model("audio", "unimodal_audio.pt")
            results["uni_audio"] = f1_a
            print("Unimodal audio F1:", f1_a, "Acc:", acc_a)
        except FileNotFoundError as e:
            print("Audio model missing:", e)

        try:
            f1_v, acc_v, lab_v, pred_v = eval_unimodal_model("video", "unimodal_video.pt")
            results["uni_video"] = f1_v
            print("Unimodal video F1:", f1_v, "Acc:", acc_v)
        except FileNotFoundError as e:
            print("Video model missing:", e)

        # Multimodal
        try:
            f1_early, acc_early, lab_early, pred_early = eval_multimodal_model("early", "multimodal_early_fusion.pt")
            results["mm_early"] = f1_early
            print("Multimodal early-fusion F1:", f1_early, "Acc:", acc_early)
        except FileNotFoundError as e:
            print("Early-fusion model missing:", e)

        try:
            f1_late, acc_late, lab_late, pred_late = eval_multimodal_model("late", "multimodal_late_fusion.pt")
            results["mm_late"] = f1_late
            print("Multimodal late-fusion F1:", f1_late, "Acc:", acc_late)
        except FileNotFoundError as e:
            print("Late-fusion model missing:", e)

        if results:
            plot_f1_bar(results, "MELD Dev Weighted F1 (Unimodal vs Multimodal)")

            # Example confusion matrix: early fusion if available, else text
            if "mm_early" in results:
                plot_confusion(lab_early, pred_early, "Early-fusion multimodal (dev)")
            elif "uni_text" in results:
                plot_confusion(lab_t, pred_t, "Unimodal text (dev)")


__all__ = [
    "Paths",
    "ModelNames",
    "TrainConfig",
    "EMOTIONS",
    "label2id",
    "id2label",
    "TextEncoder",
    "AudioEncoder",
    "VideoEncoder",
    "UtteranceDataset",
    "ConversationDataset",
    "collate_conversations",
    "UnimodalClassifier",
    "EarlyFusionContext",
    "LateFusionContext",
    "run_feature_extraction",
    "train_unimodal",
    "train_multimodal",
    "eval_unimodal_model",
    "eval_multimodal_model",
    "plot_f1_bar",
    "plot_confusion",
    "run_pipeline",
    "paths",
    "train_cfg",
    "model_names",
    "device",
]


if __name__ == "__main__":
    # Example: run everything by toggling flags here
    run_pipeline(
        do_extract_features=False,
        do_train_unimodal=False,
        do_train_mm_early=False,
        do_train_mm_late=False,
        do_plot_eval=False,
    )