#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render Mermaid diagrams to PNG for report assets (§19).

Uses mermaid.ink (pako-compressed JSON). Prefer balanced aspect ratios:
avoid very tall/narrow or very wide/short layouts when authoring diagrams.

Usage:
  python mermaid_to_png.py diagram.mmd assets/out.png
  python mermaid_to_png.py --stdin assets/out.png < diagram.mmd
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.request
import zlib
from io import BytesIO
from pathlib import Path


def encode_mermaid(diagram: str) -> str:
    payload = {"code": diagram, "mermaid": {"theme": "default"}}
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode("ascii")


def fetch_raster(diagram: str, timeout: float = 60.0) -> bytes:
    b64 = encode_mermaid(diagram)
    url = f"https://mermaid.ink/img/pako:{b64}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def to_png_rgb(raster: bytes) -> bytes:
    try:
        from PIL import Image
    except ImportError as e:
        raise SystemExit("Pillow required: pip install Pillow") from e
    img = Image.open(BytesIO(raster)).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img.split()[-1])
    out = BytesIO()
    bg.convert("RGB").save(out, "PNG", optimize=True)
    return out.getvalue()


def render(diagram: str, out_path: Path) -> None:
    diagram = diagram.strip()
    if not diagram:
        raise SystemExit("empty mermaid diagram")
    png = to_png_rgb(fetch_raster(diagram))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(png)
    print(f"wrote {out_path} ({len(png)} bytes)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Mermaid → PNG via mermaid.ink")
    ap.add_argument("input", nargs="?", type=Path, help="Input .mmd file (or --stdin)")
    ap.add_argument("output", type=Path, help="Output .png path")
    ap.add_argument("--stdin", action="store_true", help="Read diagram from stdin")
    args = ap.parse_args()
    if args.stdin:
        diagram = sys.stdin.read()
    elif args.input is not None:
        diagram = args.input.read_text(encoding="utf-8")
    else:
        ap.error("provide input .mmd or --stdin")
    render(diagram, args.output)


if __name__ == "__main__":
    main()
