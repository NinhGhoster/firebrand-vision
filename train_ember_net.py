"""
train_ember_net.py — Learned multi-frame firebrand detector (EmberNet).

Small spatio-temporal U-Net: input = 5 consecutive grayscale frames, output =
per-pixel firebrand heatmap for the CENTER frame. Motion context is in the
input stack, so the net can learn "small bright thing FLYING" vs "glow sitting
on the fuel" — the distinction hand rules approximate.

Training data comes from the vision-only auto-labels:
  positives       accepted firebrand detections (Gaussian heatmap bumps)
  hard negatives  rejected-track detections — attached glow, flame crawl,
                  floor reflections, glints (empty heatmap at their location)
  background      random patches (empty heatmap)

Split is by temporal blocks (alternating ~1-minute blocks) so val frames are
never seen in training.

Usage (Spartan):
  python train_ember_net.py --frames-dir images/frames \
      --labels-dir outputs/vision_labels --out outputs/ember_net
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vision_autolabel import list_frames  # noqa: E402

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PATCH = 192
STACK = 5  # frames per sample (center +/- 2)
SIGMA = 2.0  # heatmap bump sigma (px)
BLOCK_FRAMES = 1800  # ~1 min temporal blocks; every 5th block is validation


def is_val_frame(frame_idx):
    return (frame_idx // BLOCK_FRAMES) % 5 == 4


# ----------------------------------------------------------------------------
# Sample collection + patch cache
# ----------------------------------------------------------------------------

# rejection reasons that mark true CONFUSERS (safe negatives). Kinematic
# rejections (lifetime/displacement/consistency/speed) are often real embers
# with insufficient track evidence — those become IGNORE regions, never
# negatives, or the model is taught to suppress real embers.
CONFUSER_REASONS = {"diffuse", "on_structure", "too_dim", "below_horizon",
                    "never_exits_fuel_roi", "pre_ignition"}


def collect_samples(labels_dir, max_pos=60000, max_hardneg=20000, n_bgneg=8000,
                    frame_ids=None, rng=None):
    """Return (samples, pos_by_frame, ignore_by_frame).
    samples: list of (center_frame, cx, cy, kind). Targets are rendered later
    from pos_by_frame (ALL accepted embers near a patch get bumps);
    ignore_by_frame positions get zero loss (ambiguous detections)."""
    rng = rng or random.Random(SEED)
    pos_by_frame = {}
    with open(Path(labels_dir) / "detections_vision.jsonl") as f:
        for line in f:
            e = json.loads(line)
            if e["n_candidates"]:
                pos_by_frame[e["frame_0based"]] = [
                    (float(d["centroid"][0]), float(d["centroid"][1]))
                    for d in e["detections"]]

    samples = []
    for fr, pts in pos_by_frame.items():
        for cx, cy in pts:
            samples.append((fr, cx, cy, "pos"))
    rng.shuffle(samples)
    samples = samples[:max_pos]

    hard = []
    ignore_by_frame = {}
    rej_path = Path(labels_dir) / "rejected_detections.jsonl"
    if rej_path.exists():
        with open(rej_path) as f:
            for line in f:
                e = json.loads(line)
                pt = (float(e["centroid"][0]), float(e["centroid"][1]))
                if e.get("reason") in CONFUSER_REASONS:
                    hard.append((e["frame_0based"], pt[0], pt[1], "hardneg"))
                else:
                    ignore_by_frame.setdefault(e["frame_0based"], []).append(pt)
        rng.shuffle(hard)
        hard = hard[:max_hardneg]
        print(f"[INFO] confuser hard negatives: {len(hard)}, "
              f"ignore frames: {len(ignore_by_frame)}", flush=True)
    else:
        print("[WARN] rejected_detections.jsonl missing — no hard negatives")

    bg = []
    if frame_ids:
        for _ in range(n_bgneg):
            fr = rng.choice(frame_ids)
            bg.append((fr, rng.uniform(40, 980), rng.uniform(40, 720), "bgneg"))

    return samples + hard + bg, pos_by_frame, ignore_by_frame


def build_patch_cache(samples, frames, pos_by_frame, ignore_by_frame, cache_dir):
    """One sequential pass over the video, writing uint8 patch stacks to a
    memmap. Epochs then never touch the PNGs."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = dict(frames)
    valid_ids = set(frame_paths)
    # keep only samples whose full stack exists
    samples = [s for s in samples
               if all((s[0] + o) in valid_ids for o in range(-2, 3))]

    by_frame = {}
    for si, s in enumerate(samples):
        by_frame.setdefault(s[0], []).append(si)

    mm_path = cache_dir / "patches.npy"
    mm = np.lib.format.open_memmap(
        mm_path, mode="w+", dtype=np.uint8,
        shape=(len(samples), STACK, PATCH, PATCH))
    meta = [None] * len(samples)

    order = sorted(by_frame)
    from collections import deque
    window = deque(maxlen=STACK)
    win_ids = deque(maxlen=STACK)
    t0 = time.time()
    done = 0
    all_ids = sorted(valid_ids)
    for fi in all_ids:
        img = cv2.imread(frame_paths[fi], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((768, 1024), np.uint8)
        window.append(img)
        win_ids.append(fi)
        center = fi - 2
        if len(window) < STACK or center not in by_frame:
            continue
        stack = np.stack(window)  # frames center-2 .. center+2
        h, w = img.shape
        for si in by_frame[center]:
            fr, cx, cy, kind = samples[si]
            x0 = int(np.clip(cx - PATCH // 2, 0, w - PATCH))
            y0 = int(np.clip(cy - PATCH // 2, 0, h - PATCH))
            mm[si] = stack[:, y0:y0 + PATCH, x0:x0 + PATCH]
            bumps = [(px - x0, py - y0)
                     for px, py in pos_by_frame.get(center, [])
                     if x0 - 8 <= px < x0 + PATCH + 8 and y0 - 8 <= py < y0 + PATCH + 8]
            ignores = [(px - x0, py - y0)
                       for px, py in ignore_by_frame.get(center, [])
                       if x0 - 8 <= px < x0 + PATCH + 8 and y0 - 8 <= py < y0 + PATCH + 8]
            meta[si] = {"kind": kind, "frame": center, "bumps": bumps,
                        "ignores": ignores, "val": is_val_frame(center)}
            done += 1
        if done and done % 20000 == 0:
            print(f"[INFO] cache: {done}/{len(samples)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    mm.flush()
    keep = [i for i, m in enumerate(meta) if m is not None]
    with open(cache_dir / "meta.json", "w") as f:
        json.dump({"meta": [meta[i] for i in keep], "rows": keep,
                   "n_total": len(samples)}, f)
    print(f"[INFO] cache built: {len(keep)} samples in {time.time() - t0:.0f}s",
          flush=True)


def _gauss_add(img, cx, cy, sigma, peak):
    """Additive Gaussian point source on a float image (max-composited)."""
    h, w = img.shape
    r = int(max(3, sigma * 4))
    x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r + 1)
    y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    g = peak * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    np.maximum(img[y0:y1, x0:x1], g.astype(img.dtype), out=img[y0:y1, x0:x1])


def composite_synthetic_embers(stack, rng, n_max=3):
    """Composite 1..n_max synthetic embers onto a REAL 5-frame patch stack
    (float [0,1]): ballistic trajectories, slow-shutter motion-blur streaks,
    brightness decay. Returns center-frame bump positions — unlimited exact
    positives on real cluttered backgrounds."""
    h, w = stack.shape[1:]
    bumps = []
    for _ in range(int(rng.integers(1, n_max + 1))):
        x0, y0 = rng.uniform(12, w - 12), rng.uniform(12, h - 12)
        speed = rng.uniform(1.5, 28.0)
        ang = rng.uniform(0, 2 * np.pi)
        vx, vy = speed * np.cos(ang), speed * np.sin(ang)
        sigma = rng.uniform(1.0, 2.6)
        peak = rng.uniform(0.20, 0.75)   # gray ~50-190: dim to bright embers
        fade = rng.uniform(0.88, 1.0)    # cooling across the 5 frames
        for t in range(STACK):
            cx, cy = x0 + vx * (t - 2), y0 + vy * (t - 2)
            pk = peak * (fade ** t)
            nsub = max(1, int(min(speed, 24) // 3))  # streak length ~ speed
            for s in range(nsub):
                f = (s + 0.5) / nsub - 0.5
                _gauss_add(stack[t], cx + vx * f, cy + vy * f, sigma,
                           pk * (0.75 if nsub > 1 else 1.0))
        if 6 <= x0 < w - 6 and 6 <= y0 < h - 6:
            bumps.append((float(x0), float(y0)))
    return stack, bumps


class PatchDataset(Dataset):
    def __init__(self, cache_dir, val=False, augment=True, synth_frac=0.0):
        cache_dir = Path(cache_dir)
        info = json.load(open(cache_dir / "meta.json"))
        self.mm = np.load(cache_dir / "patches.npy", mmap_mode="r")
        pairs = [(r, m) for r, m in zip(info["rows"], info["meta"])
                 if m["val"] == val]
        self.rows = [r for r, _ in pairs]
        self.meta = [m for _, m in pairs]
        self.augment = augment and not val
        self.synth_frac = synth_frac if self.augment else 0.0
        self.rng = np.random.default_rng(SEED)

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def render_heatmap(bumps, flip=False):
        hm = np.zeros((PATCH, PATCH), np.float32)
        for bx, by in bumps:
            if flip:
                bx = PATCH - 1 - bx
            x0, x1 = int(max(0, bx - 8)), int(min(PATCH, bx + 9))
            y0, y1 = int(max(0, by - 8)), int(min(PATCH, by + 9))
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            g = np.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / (2 * SIGMA ** 2))
            hm[y0:y1, x0:x1] = np.maximum(hm[y0:y1, x0:x1], g)
        return hm

    @staticmethod
    def render_loss_mask(ignores, flip=False, radius=7):
        """1 where loss applies; 0 around ambiguous (kinematically-rejected)
        detections — often real embers with too little track evidence."""
        mask = np.ones((PATCH, PATCH), np.float32)
        for bx, by in ignores:
            if flip:
                bx = PATCH - 1 - bx
            cv2.circle(mask, (int(bx), int(by)), radius, 0.0, -1)
        return mask

    def __getitem__(self, i):
        x = self.mm[self.rows[i]].astype(np.float32) / 255.0
        m = self.meta[i]
        bumps = list(m["bumps"])
        if self.synth_frac and random.random() < self.synth_frac:
            x = np.ascontiguousarray(x)
            x, extra = composite_synthetic_embers(x, self.rng)
            bumps += extra
        flip = self.augment and random.random() < 0.5
        if flip:
            x = x[:, :, ::-1].copy()
        if self.augment:
            x = np.clip(x * random.uniform(0.9, 1.1), 0, 1)
        # explicit motion channel: center frame minus 5-frame median (the
        # median erases the transient ember, so the ember pops in the diff)
        med = np.median(x, axis=0)
        diff = np.clip(x[STACK // 2] - med, 0, 1)[None]
        x = np.concatenate([x, diff], axis=0)
        hm = self.render_heatmap(bumps, flip=flip)
        mask = self.render_loss_mask(m.get("ignores", []), flip=flip)
        return (torch.from_numpy(x), torch.from_numpy(hm[None]),
                torch.from_numpy(mask[None]))


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class EmberNet(nn.Module):
    def __init__(self, cin=STACK + 1, base=16):  # 5 frames + motion-diff channel
        super().__init__()
        b = base
        self.e1 = conv_block(cin, b)
        self.e2 = conv_block(b, b * 2)
        self.e3 = conv_block(b * 2, b * 4)
        self.e4 = conv_block(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.d3 = conv_block(b * 8, b * 4)
        self.u2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.d2 = conv_block(b * 4, b * 2)
        self.u1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.d1 = conv_block(b * 2, b)
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(e4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.head(d1)


def focal_loss(logits, gt, mask=None, alpha=2.0, beta=4.0):
    """CenterNet penalty-reduced focal loss on Gaussian heatmaps.
    mask: 1 where loss applies, 0 over ignore regions."""
    pred = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = gt.ge(0.99).float()
    neg_w = (1 - gt).pow(beta)
    pos_loss = -((1 - pred).pow(alpha)) * torch.log(pred) * pos
    neg_loss = -(pred.pow(alpha)) * torch.log(1 - pred) * neg_w * (1 - pos)
    if mask is not None:
        pos_loss = pos_loss * mask
        neg_loss = neg_loss * mask
    npos = (pos * (mask if mask is not None else 1)).sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / npos


# ----------------------------------------------------------------------------
# Peak extraction + val detection metric
# ----------------------------------------------------------------------------

def extract_peaks(heat, thresh=0.30):
    """heat: (H, W) tensor -> [(x, y, score)] local maxima above thresh."""
    hmax = F.max_pool2d(heat[None, None], 5, stride=1, padding=2)[0, 0]
    mask = (heat == hmax) & (heat >= thresh)
    ys, xs = torch.nonzero(mask, as_tuple=True)
    return [(int(x), int(y), float(heat[y, x])) for y, x in zip(ys, xs)]


@torch.no_grad()
def val_detection_f1(model, loader, device, match_px=6,
                     threshes=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4)):
    """Sweep detection thresholds; return the best (P, R, F1, thresh) so
    mis-calibration cannot hide real progress."""
    model.eval()
    stats = {t: [0, 0, 0] for t in threshes}  # tp, fp, fn
    for x, hm, _ in loader:
        x = x.to(device, non_blocking=True)
        pred = torch.sigmoid(model(x)).cpu()
        for b in range(x.shape[0]):
            gt_peaks = extract_peaks(hm[b, 0], 0.99)
            for t in threshes:
                peaks = extract_peaks(pred[b, 0], t)
                used = set()
                tp = fp = 0
                for px, py, _ in peaks:
                    hit = None
                    for gi, (gx, gy, _) in enumerate(gt_peaks):
                        if gi not in used and (px - gx) ** 2 + (py - gy) ** 2 <= match_px ** 2:
                            hit = gi
                            break
                    if hit is None:
                        fp += 1
                    else:
                        used.add(hit)
                        tp += 1
                stats[t][0] += tp
                stats[t][1] += fp
                stats[t][2] += len(gt_peaks) - len(used)
    best = (0.0, 0.0, 0.0, threshes[0])
    for t, (tp, fp, fn) in stats.items():
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best[2]:
            best = (prec, rec, f1, t)
    return best


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train EmberNet")
    parser.add_argument("--frames-dir", default="images/frames")
    parser.add_argument("--labels-dir", default="outputs/vision_labels")
    parser.add_argument("--out", default="outputs/ember_net")
    parser.add_argument("--cache-dir", default=None,
                        help="default: <out>/cache")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-pos", type=int, default=60000)
    parser.add_argument("--base-width", type=int, default=16,
                        help="EmberNet channel width (32 for the H100 config)")
    parser.add_argument("--synth-frac", type=float, default=0.0,
                        help="fraction of train patches composited with "
                             "synthetic embers (e.g. 0.5)")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir or out / "cache")

    if args.rebuild_cache or not (cache_dir / "meta.json").exists():
        frames = list_frames(args.frames_dir, 0, -1)
        frame_ids = [i for i, _ in frames]
        samples, pos_by_frame, ignore_by_frame = collect_samples(
            args.labels_dir, max_pos=args.max_pos, frame_ids=frame_ids)
        print(f"[INFO] samples: {len(samples)}", flush=True)
        build_patch_cache(samples, frames, pos_by_frame, ignore_by_frame,
                          cache_dir)

    train_ds = PatchDataset(cache_dir, val=False, synth_frac=args.synth_frac)
    val_ds = PatchDataset(cache_dir, val=True, augment=False)
    print(f"[INFO] train {len(train_ds)} / val {len(val_ds)}", flush=True)

    # oversample patches that contain embers (most patches are negatives)
    weights = [3.0 if m["bumps"] else 1.0 for m in train_ds.meta]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}", flush=True)
    model = EmberNet(base=args.base_width).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] EmberNet params: {n_params/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        tl, nb = 0.0, 0
        t0 = time.time()
        for x, hm, mask in train_loader:
            x = x.to(device, non_blocking=True)
            hm = hm.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = focal_loss(model(x), hm, mask)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.item()
            nb += 1
        sched.step()

        prec, rec, f1, th = val_detection_f1(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": tl / max(nb, 1),
                        "val_precision": round(prec, 4),
                        "val_recall": round(rec, 4), "val_f1": round(f1, 4),
                        "val_thresh": th})
        print(f"[E{epoch:02d}] loss {tl/max(nb,1):.4f} | "
              f"val P {prec:.3f} R {rec:.3f} F1 {f1:.3f} @th={th} | "
              f"{time.time()-t0:.0f}s", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model": model.state_dict(), "f1": f1, "thresh": th,
                        "epoch": epoch, "base_width": args.base_width},
                       out / "best.pt")
            print(f"[INFO] saved best (F1={f1:.3f})", flush=True)

    torch.save({"model": model.state_dict()}, out / "final.pt")
    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=1)
    print(f"[INFO] done. best val F1: {best_f1:.3f}", flush=True)


if __name__ == "__main__":
    main()
