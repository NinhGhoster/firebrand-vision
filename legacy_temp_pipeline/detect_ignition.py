import argparse
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Detect ignition frame from thermal NC data")
    parser.add_argument("--nc", default="S12.nc", help="Input NC file path")
    parser.add_argument("--threshold", type=float, default=200.0,
                        help="Temperature threshold for ignition (°C)")
    parser.add_argument("--min-frames", type=int, default=3,
                        help="Consecutive frames above threshold to confirm ignition")
    parser.add_argument("--out", default="outputs/ignition.json",
                        help="Output JSON file")
    args = parser.parse_args()

    print(f"[INFO] Loading {args.nc}...")
    t0 = time.time()

    try:
        import xarray as xr
        ds = xr.open_dataset(args.nc, engine="netcdf4")
    except Exception:
        try:
            import netCDF4 as nc4
            ds = nc4.Dataset(args.nc)
            temp = ds.variables["temperature"][:]
            ds.close()
            _detect(temp, args)
            return
        except Exception as e2:
            print(f"[ERROR] Cannot open {args.nc}: {e2}")
            return

    temp = ds.temperature.values
    ds.close()

    elapsed = time.time() - t0
    print(f"[INFO] Loaded {temp.shape[0]} frames, shape {temp.shape[1:]} in {elapsed:.1f}s")

    _detect(temp, args)


def _detect(temp, args):
    print(f"[INFO] Scanning for first frame >{args.threshold}°C...")

    consecutive = 0
    ignition_frame = None

    for i in range(temp.shape[0]):
        max_t = float(temp[i].max())
        if max_t > args.threshold:
            consecutive += 1
            if consecutive >= args.min_frames:
                ignition_frame = i - args.min_frames + 1
                break
        else:
            consecutive = 0

    if ignition_frame is not None:
        t_s = ignition_frame / 29.41
        print(f"\n=== Ignition Detected ===")
        print(f"Ignition frame (0-based): {ignition_frame}")
        print(f"Ignition time: {t_s:.2f}s")
        print(f"Frame number (1-based): {ignition_frame + 1}")

        # Report max temp at ignition
        if hasattr(args, 'nc_detail'):
            pass
        else:
            max_at_ignition = float(temp[ignition_frame].max())
            print(f"Max temp at ignition: {max_at_ignition:.1f}°C")
    else:
        ignition_frame = None
        print(f"[WARN] No ignition detected (threshold {args.threshold}°C not reached)")

    # Save JSON
    result = {
        "ignition_frame_0based": ignition_frame,
        "ignition_time_s": round(ignition_frame / 29.41, 2) if ignition_frame is not None else None,
        "threshold_degC": args.threshold,
        "total_frames": temp.shape[0],
    }
    from pathlib import Path
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] Saved ignition data to {args.out}")


if __name__ == "__main__":
    main()
