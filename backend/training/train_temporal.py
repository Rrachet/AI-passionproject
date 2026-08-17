"""Train a small temporal gesture model from feedback.jsonl.

This is intentionally separate from the production API. It learns from 21 hand
landmarks over short windows instead of retraining the hand detector itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

LABELS = {"DRAW": 0, "PAUSE": 1, "CLEAR": 2, "UNKNOWN": 3}
WINDOW = 24
FEATURES = 63


class SequenceDataset(Dataset):
    def __init__(self, path: str):
        rows = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                label = row.get("corrected") or row.get("predicted")
                landmarks = row.get("landmarks", [])
                if label not in LABELS or len(landmarks) != 21:
                    continue
                flat = []
                for p in landmarks:
                    flat.extend([float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("z", 0))])
                rows.append((flat, LABELS[label]))
        if len(rows) < WINDOW:
            raise ValueError(f"Need at least {WINDOW} labeled samples; found {len(rows)}")
        self.x = torch.tensor([r[0] for r in rows], dtype=torch.float32)
        self.y = torch.tensor([r[1] for r in rows], dtype=torch.long)

    def __len__(self):
        return max(1, len(self.x) - WINDOW + 1)

    def __getitem__(self, i):
        end = i + WINDOW
        return self.x[i:end], self.y[end - 1]


class AirCanvasGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(FEATURES, 96, num_layers=2, batch_first=True, dropout=0.15)
        self.head = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 64), nn.GELU(), nn.Linear(64, len(LABELS)))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1])


def train(data_path="backend/data/feedback.jsonl", epochs=25, batch_size=32):
    ds = SequenceDataset(data_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = AirCanvasGRU()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss)
        print(f"epoch={epoch + 1:02d} loss={total / len(loader):.4f}")
    out = Path("backend/models/aircanvas-gru.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "labels": LABELS, "window": WINDOW}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    train()
