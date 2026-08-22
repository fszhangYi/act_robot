#!/usr/bin/env python3
"""PDF → Markdown scaffold for research papers (PyMuPDF).

Extracts page text, optionally dumps embedded images / page renders /
caption hit boxes / manual crops. Output MD is a draft for the agent to
structure (tables, equations, bilingual pass) — not a finished paper.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _require_fitz():
    try:
        import pymupdf as fitz  # noqa: F401
        return fitz
    except ImportError:
        try:
            import fitz  # type: ignore
            return fitz
        except ImportError as e:
            raise SystemExit(
                "Need pymupdf: pip install pymupdf"
            ) from e


CAPTION_PATTERNS = (
    "Fig. ",
    "Fig.",
    "Figure ",
    "FIGURE ",
    "TABLE ",
    "Table ",
)


def extract_text_pages(doc) -> list[str]:
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        pages.append(text.rstrip() + "\n")
    return pages


def write_scaffold_md(
    out_md: Path,
    pdf_path: Path,
    pages: list[str],
    assets_rel: str,
    image_names: list[str],
) -> None:
    stem = pdf_path.stem
    lines: list[str] = [
        f"# {stem}",
        "",
        f"**Source PDF:** [`{pdf_path.name}`]({pdf_path.name})",
        "",
        "> Draft from `pdf_to_md.py`. Restructure headings/tables/equations; "
        "replace placeholders with cropped figures as needed.",
        "",
    ]
    if image_names:
        lines.append("## Extracted embedded images")
        lines.append("")
        for name in image_names:
            rel = f"{assets_rel}/{name}".replace("\\", "/")
            lines.append(f"- `{rel}`")
            lines.append(f"  ![{name}]({rel})")
            lines.append("")
    lines.append("## Page text (raw)")
    lines.append("")
    for i, text in enumerate(pages, start=1):
        lines.append(f"### Page {i}")
        lines.append("")
        lines.append("```")
        lines.append(text.rstrip())
        lines.append("```")
        lines.append("")
        lines.append(
            f"> **【图注占位】** If this page has figures, export to "
            f"`{assets_rel}/fig….png` and replace this note."
        )
        lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def extract_images(doc, assets_dir: Path) -> list[str]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    seen: set[int] = set()
    for page_index, page in enumerate(doc):
        for img_i, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            if xref in seen:
                continue
            seen.add(xref)
            try:
                import pymupdf as fitz
            except ImportError:
                import fitz  # type: ignore
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha > 3:  # CMYK etc.
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                name = f"embed_p{page_index + 1:02d}_{img_i:02d}_x{xref}.png"
                path = assets_dir / name
                pix.save(str(path))
                names.append(name)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] skip xref={xref}: {exc}", file=sys.stderr)
    return names


def render_pages(doc, assets_dir: Path, dpi: float) -> list[str]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz  # type: ignore
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    names = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        name = f"page_{i + 1:02d}.png"
        pix.save(str(assets_dir / name))
        names.append(name)
        print(f"[page] {name} {pix.width}x{pix.height}")
    return names


def list_captions(doc) -> None:
    for i, page in enumerate(doc):
        for pat in CAPTION_PATTERNS:
            for hit in page.search_for(pat):
                print(f"page {i} ({i + 1}-based {i + 1}) {pat!r} @ {hit}")


def parse_crop(spec: str):
    """page_index,name,x0,y0,x1,y1"""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 6:
        raise SystemExit(f"bad --crop {spec!r}; want page,name,x0,y0,x1,y1")
    page_i = int(parts[0])
    name = parts[1]
    box = tuple(float(x) for x in parts[2:])
    return page_i, name, box


def apply_crops(doc, assets_dir: Path, crops: list[str], dpi: float) -> list[str]:
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz  # type: ignore
    assets_dir.mkdir(parents=True, exist_ok=True)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    names = []
    for spec in crops:
        page_i, name, (x0, y0, x1, y1) = parse_crop(spec)
        if page_i < 0 or page_i >= len(doc):
            raise SystemExit(f"crop page_index out of range: {page_i}")
        page = doc[page_i]
        clip = fitz.Rect(x0, y0, x1, y1)
        pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
        out_name = name if name.endswith(".png") else f"{name}.png"
        pix.save(str(assets_dir / out_name))
        names.append(out_name)
        print(f"[crop] {out_name} page={page_i} clip={clip} {pix.width}x{pix.height}")
    return names


def assets_rel_for_md(md_path: Path, assets_dir: Path) -> str:
    try:
        rel = assets_dir.resolve().relative_to(md_path.parent.resolve())
        return str(rel).replace("\\", "/")
    except ValueError:
        return str(assets_dir).replace("\\", "/")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", type=Path, help="Input PDF")
    ap.add_argument(
        "md",
        type=Path,
        nargs="?",
        default=None,
        help="Output Markdown (default: same stem next to PDF)",
    )
    ap.add_argument(
        "--assets-dir",
        type=Path,
        default=None,
        help="Directory for images (default: <md_parent>/assets/<stem>)",
    )
    ap.add_argument("--extract-images", action="store_true", help="Dump embedded images")
    ap.add_argument("--render-pages", action="store_true", help="Render full pages for inspection")
    ap.add_argument("--list-captions", action="store_true", help="Print Fig./Table hit boxes")
    ap.add_argument(
        "--crop",
        action="append",
        default=[],
        metavar="SPEC",
        help="Crop: page_index,name,x0,y0,x1,y1 (repeatable)",
    )
    ap.add_argument("--dpi", type=float, default=216.0, help="Render/crop DPI (default 216)")
    ap.add_argument(
        "--no-md",
        action="store_true",
        help="Only extract/render/crop; do not write Markdown scaffold",
    )
    args = ap.parse_args(argv)

    pdf_path = args.pdf.expanduser().resolve()
    if not pdf_path.is_file():
        raise SystemExit(f"PDF not found: {pdf_path}")

    md_path = (
        args.md.expanduser().resolve()
        if args.md
        else pdf_path.with_suffix(".md")
    )
    assets_dir = (
        args.assets_dir.expanduser().resolve()
        if args.assets_dir
        else (md_path.parent / "assets" / pdf_path.stem).resolve()
    )

    fitz = _require_fitz()
    doc = fitz.open(pdf_path)
    print(f"pages={len(doc)} pdf={pdf_path}")

    image_names: list[str] = []
    if args.extract_images:
        image_names = extract_images(doc, assets_dir)
        print(f"[images] extracted {len(image_names)} -> {assets_dir}")

    if args.render_pages:
        render_pages(doc, assets_dir, args.dpi)

    if args.list_captions:
        list_captions(doc)

    if args.crop:
        apply_crops(doc, assets_dir, args.crop, args.dpi)

    if not args.no_md:
        pages = extract_text_pages(doc)
        rel = assets_rel_for_md(md_path, assets_dir)
        write_scaffold_md(md_path, pdf_path, pages, rel, image_names)
        print(f"wrote scaffold {md_path}")

    doc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
