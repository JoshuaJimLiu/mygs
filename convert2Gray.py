#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def iter_image_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def is_already_gray(img: Image.Image) -> bool:
    if img.mode in ("L", "LA", "1"):
        return True
    try:
        rgb = img.convert("RGB")
        w, h = rgb.size
        if w == 0 or h == 0:
            return False
        samples = [
            (0, 0),
            (w - 1, 0),
            (0, h - 1),
            (w - 1, h - 1),
            (w // 2, h // 2),
        ]
        for x, y in samples:
            r, g, b = rgb.getpixel((x, y))
            if not (r == g == b):
                return False
        return True
    except Exception:
        return False


def make_tmp_path(path: Path) -> Path:
    # 关键：保持原扩展名不变，避免 Pillow 按 .tmp 识别格式而 ValueError
    return path.with_name(f"{path.stem}.__tmp__{path.suffix}")


def convert_in_place(path: Path, make_backup: bool, dry_run: bool) -> str:
    try:
        suffix = path.suffix.lower()

        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)

            if is_already_gray(img):
                return "skip(gray)"

            has_alpha = "A" in img.getbands()

            # 生成灰度，尽量保留 alpha（但 JPEG 不支持 alpha）
            if has_alpha and suffix not in (".jpg", ".jpeg"):
                rgba = img.convert("RGBA")
                r, g, b, a = rgba.split()
                gray = Image.merge("RGB", (r, g, b)).convert("L")
                out = Image.merge("LA", (gray, a))  # PNG/WebP/TIFF 可用
            else:
                # JPEG/或无 alpha：直接 L
                out = img.convert("L")

        if dry_run:
            return "dryrun"

        if make_backup:
            bak = path.with_name(path.name + ".bak")
            if not bak.exists():
                shutil.copy2(path, bak)

        tmp = make_tmp_path(path)

        save_kwargs = {}
        if suffix in (".jpg", ".jpeg"):
            save_kwargs.update(dict(quality=95, subsampling=0, optimize=True))

        out.save(tmp, **save_kwargs)
        os.replace(tmp, path)

        return "ok"

    except UnidentifiedImageError:
        return "skip(unidentified)"
    except Exception as e:
        return f"err({type(e).__name__}: {e})"


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert images to grayscale in-place (overwrite).")
    ap.add_argument("dir", nargs="?", default=".", help="Target directory (default: current).")
    ap.add_argument("--backup", action="store_true", help="Create *.bak backups before overwriting.")
    ap.add_argument("--dry-run", action="store_true", help="Scan and report without writing files.")
    args = ap.parse_args()

    root = Path(args.dir).resolve()
    if not root.exists() or not root.is_dir():
        print(f"[ERROR] Not a directory: {root}", file=sys.stderr)
        return 2

    total = ok = skipped = errors = 0

    for p in iter_image_files(root):
        total += 1
        status = convert_in_place(p, make_backup=args.backup, dry_run=args.dry_run)
        if status == "ok":
            ok += 1
        elif status.startswith("skip"):
            skipped += 1
        else:
            errors += 1
        print(f"{status:20s} {p}")

    print("\n=== Summary ===")
    print(f"Root     : {root}")
    print(f"Total    : {total}")
    print(f"Converted: {ok}")
    print(f"Skipped  : {skipped}")
    print(f"Errors   : {errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
