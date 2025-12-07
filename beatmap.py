import os
import zipfile

def extract_osz(osz_path, target_dir="beatmaps"):
    """Extract a .osz file (osu beatmap package) into beatmaps directory."""
    with zipfile.ZipFile(osz_path, 'r') as zf:
        folder_name = os.path.splitext(os.path.basename(osz_path))[0]
        out_dir = os.path.join(target_dir, folder_name)
        os.makedirs(out_dir, exist_ok=True)
        zf.extractall(out_dir)
    print(f"Extracted to {out_dir}")
    return out_dir


def parse_osu_file(file_path):
    """Parse a .osu beatmap file and return list of (x, y, time) hit objects."""
    hit_objects = []
    parsing = False
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "[HitObjects]":
                parsing = True
                continue
            if parsing and line:
                parts = line.split(",")
                if len(parts) >= 3:
                    x, y, time = int(parts[0]), int(parts[1]), int(parts[2])
                    hit_objects.append((x, y, time))
    return hit_objects
