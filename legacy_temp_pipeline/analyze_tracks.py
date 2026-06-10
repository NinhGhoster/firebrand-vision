#!/usr/bin/env python3
import json

with open("/data/gpfs/projects/punim1990/locateanything/outputs/tracks_refined.json") as f:
    data = json.load(f)

long = [t for t in data["tracks"] if t["lifetime_frames"] >= 100]
print(f"Long tracks (>=100 frames): {len(long)}")
for t in sorted(long, key=lambda x: x["lifetime_frames"], reverse=True)[:10]:
    print(f"  Track {t['track_id']}: {t['lifetime_frames']}f ({t['lifetime_s']:.1f}s)"
          f" speed={t['avg_speed_px_per_frame']:.2f}"
          f" start=({t['start_centroid'][0]},{t['start_centroid'][1]})"
          f" end=({t['end_centroid'][0]},{t['end_centroid'][1]})")

short = [t for t in data["tracks"] if t["lifetime_frames"] < 10]
stationary = sum(1 for t in short if t["avg_speed_px_per_frame"] == 0)
print(f"\nShort tracks (<10f): total={len(short)} stationary={stationary}")

medium = [t for t in data["tracks"] if 10 <= t["lifetime_frames"] < 100]
print(f"Medium tracks (10-99f): {len(medium)}")
if medium:
    ms = [t["avg_speed_px_per_frame"] for t in medium if t["avg_speed_px_per_frame"] > 0]
    print(f"  Mean speed: {sum(ms)/len(ms):.2f} px/frame" if ms else "  All stationary")
