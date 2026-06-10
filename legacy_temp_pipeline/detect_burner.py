import argparse
import time
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Detect burner region from thermal NC data")
    parser.add_argument("--nc", default="S12.nc", help="Input NC file path")
    parser.add_argument("--temp-threshold", type=float, default=200.0,
                        help="Temperature threshold to consider 'hot' (°C)")
    parser.add_argument("--frequency-threshold", type=float, default=0.5,
                        help="Min fraction of frames a pixel must be hot to be burner")
    parser.add_argument("--ignition-frame", type=int, default=None,
                        help="Pre-known ignition frame (0-based). If unset, auto-detect.")
    parser.add_argument("--analysis-window", type=int, default=300,
                        help="Number of frames after ignition to analyze for burner")
    parser.add_argument("--min-area", type=int, default=500,
                        help="Minimum connected-component area to be burner")
    parser.add_argument("--out", default="outputs/burner_mask.json",
                        help="Output JSON with burner mask parameters")
    args = parser.parse_args()

    print(f"[INFO] Loading {args.nc}...")
    t0 = time.time()

    try:
        import xarray as xr
        ds = xr.open_dataset(args.nc, engine="netcdf4")
        temp = ds.temperature.values
        ds.close()
    except Exception:
        try:
            import netCDF4 as nc4
            ds = nc4.Dataset(args.nc)
            temp = ds.variables["temperature"][:]
            ds.close()
        except Exception as e2:
            print(f"[ERROR] Cannot open {args.nc}: {e2}")
            return

    elapsed = time.time() - t0
    n_frames, frame_h, frame_w = temp.shape
    print(f"[INFO] Loaded {n_frames} frames ({frame_w}x{frame_h}) in {elapsed:.1f}s")

    # Find ignition frame if not provided
    if args.ignition_frame is None:
        print("[INFO] Auto-detecting ignition...")
        consecutive = 0
        for i in range(n_frames):
            if float(temp[i].max()) > args.temp_threshold:
                consecutive += 1
                if consecutive >= 3:
                    args.ignition_frame = i - 2
                    break
            else:
                consecutive = 0
        if args.ignition_frame is None:
            print("[ERROR] Could not detect ignition. Provide --ignition-frame manually.")
            return
        t_s = args.ignition_frame / 29.41
        print(f"[INFO] Ignition at frame {args.ignition_frame} (t={t_s:.1f}s)")

    # Analyze frames after ignition
    start = args.ignition_frame
    end = min(start + args.analysis_window, n_frames)
    print(f"[INFO] Analyzing burner region from frame {start} to {end}...")

    # Build hot-pixel frequency map
    hot_mask = np.zeros((frame_h, frame_w), dtype=np.float32)
    analyzed = 0
    for i in range(start, end):
        frame_data = temp[i]
        hot = (frame_data > args.temp_threshold).astype(np.float32)
        hot_mask += hot
        analyzed += 1

    hot_mask /= analyzed  # fraction of frames each pixel is hot

    # Find persistent hot regions
    persistent = (hot_mask >= args.frequency_threshold).astype(np.uint8) * 255

    n_comp, labels, stats, centroids = cv2.connectedComponentsWithStats(persistent, connectivity=8)

    burner_components = []
    for comp_id in range(1, n_comp):  # skip background (0)
        area = stats[comp_id, cv2.CC_STAT_AREA]
        if area >= args.min_area:
            x = stats[comp_id, cv2.CC_STAT_LEFT]
            y = stats[comp_id, cv2.CC_STAT_TOP]
            w = stats[comp_id, cv2.CC_STAT_WIDTH]
            h = stats[comp_id, cv2.CC_STAT_HEIGHT]
            cx, cy = int(centroids[comp_id][0]), int(centroids[comp_id][1])
            mean_temp = float(np.mean(temp[start:end,
                                        max(0, cy - h // 2):cy + h // 2 + 1,
                                        max(0, cx - w // 2):cx + w // 2 + 1]))
            burner_components.append({
                "component_id": int(comp_id),
                "area_px": int(area),
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "centroid": [float(round(cx, 1)), float(round(cy, 1))],
                "mean_temp": round(mean_temp, 1),
                "hot_frequency": float(round(float(hot_mask[labels == comp_id].mean()), 3)),
            })

    burner_components.sort(key=lambda c: -c["area_px"])

    print(f"\n=== Burner Detection ===")
    print(f"Frames analyzed: {analyzed}")
    print(f"Burner components found: {len(burner_components)}")
    for bc in burner_components[:3]:
        print(f"  Area {bc['area_px']}px at ({bc['centroid'][0]:.0f},{bc['centroid'][1]:.0f}) "
              f"bbox={bc['bbox']} freq={bc['hot_frequency']:.2f}")

    # Build combined mask of burner regions
    burner_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    for bc in burner_components:
        x1, y1, x2, y2 = bc["bbox"]
        # Add margin
        margin = 20
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame_w, x2 + margin)
        y2 = min(frame_h, y2 + margin)
        burner_mask[y1:y2, x1:x2] = 255

    # Save results
    result = {
        "ignition_frame_0based": args.ignition_frame,
        "ignition_time_s": round(args.ignition_frame / 29.41, 2),
        "burner_components": burner_components,
        "burner_mask_bbox": [int(burner_mask.min() == 0 or burner_mask.max() == 255),
                             burner_components[0]["bbox"] if burner_components else None],
        "image_width": frame_w,
        "image_height": frame_h,
    }

    import json
    from pathlib import Path
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[INFO] Burner mask saved to {args.out}")
    print(f"[INFO] Burner region(s): {len(burner_components)} components, "
          f"largest at ({burner_components[0]['centroid'][0]:.0f},{burner_components[0]['centroid'][1]:.0f})"
          if burner_components else "")


if __name__ == "__main__":
    main()
