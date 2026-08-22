#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DOCX → Markdown converter (inverse of md-to-docx-report).

Usage:
  python docx_to_md.py INPUT.docx [OUTPUT.md]

If OUTPUT is omitted, write INPUT with a .md suffix next to the docx.

Rules:
  - Restore Word hyperlinks as Markdown [text](url)
  - Skip TOC styles and the 目录 label
  - Skip drawing-only paragraphs (no binary image export)
  - Keep 【插图地址】 / 图： / 示意： as text
  - Heading map matches md-to-docx-report round-trip
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

WP_DRAWING = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"


def local(tag: str) -> str:
    return tag.split("}")[-1] if tag and "}" in tag else (tag or "")


def is_drawing_only(p: Paragraph) -> bool:
    el = p._element
    has_drawing = bool(
        el.findall(f".//{WP_DRAWING}inline") or el.findall(f".//{WP_DRAWING}anchor")
    )
    if not has_drawing:
        return False
    return not (p.text or "").strip()


def _on_off(el) -> bool:
    if el is None:
        return False
    val = el.get(qn("w:val"))
    if val is None:
        return True
    return val not in ("0", "false", "False")


def run_is_bold(r_el) -> bool:
    rPr = r_el.find(qn("w:rPr"))
    return _on_off(rPr.find(qn("w:b")) if rPr is not None else None)


def run_is_italic(r_el) -> bool:
    rPr = r_el.find(qn("w:rPr"))
    return _on_off(rPr.find(qn("w:i")) if rPr is not None else None)


def parse_hyperlink_instr(instr: str) -> str | None:
    instr = html.unescape(instr).strip()
    m = re.match(r'^HYPERLINK\s+"([^"]+)"', instr, re.I)
    if m:
        return m.group(1)
    m = re.match(r"^HYPERLINK\s+'([^']+)'", instr, re.I)
    if m:
        return m.group(1)
    return None


def wrap_format(text: str, bold: bool, italic: bool) -> str:
    if not text or not text.strip():
        return text
    lead = re.match(r"^\s*", text).group(0)
    trail = re.search(r"\s*$", text).group(0)
    core = text[len(lead) : len(text) - len(trail) if trail else len(text)]
    if not core:
        return text
    if bold and italic:
        core = f"***{core}***"
    elif bold:
        core = f"**{core}**"
    elif italic:
        core = f"*{core}*"
    return lead + core + trail


class FieldState:
    def __init__(self):
        self.stack: list[dict] = []

    def begin(self):
        self.stack.append({"instr": "", "parts": [], "mode": "instr"})

    def separate(self):
        if self.stack:
            self.stack[-1]["mode"] = "display"

    def end(self) -> str:
        if not self.stack:
            return ""
        frame = self.stack.pop()
        display = "".join(frame["parts"])
        url = parse_hyperlink_instr(frame["instr"]) if frame["instr"] else None
        if url:
            return f"[{display.strip() or url}]({url})"
        return display

    def add_instr(self, text: str):
        if self.stack and self.stack[-1]["mode"] == "instr":
            self.stack[-1]["instr"] += text

    def add_display(self, text: str):
        if self.stack and self.stack[-1]["mode"] == "display":
            self.stack[-1]["parts"].append(text)

    @property
    def active(self) -> bool:
        return bool(self.stack)


def paragraph_to_md(p: Paragraph) -> str:
    fields = FieldState()
    out: list[str] = []
    buf_text = ""
    buf_bold = False
    buf_italic = False

    def flush_buf():
        nonlocal buf_text, buf_bold, buf_italic
        if buf_text:
            out.append(wrap_format(buf_text, buf_bold, buf_italic))
            buf_text = ""

    def emit_plain(text: str, bold: bool = False, italic: bool = False):
        nonlocal buf_text, buf_bold, buf_italic
        if fields.active:
            flush_buf()
            fields.add_display(text)
            return
        if buf_text and (bold, italic) != (buf_bold, buf_italic):
            flush_buf()
        if not buf_text:
            buf_bold, buf_italic = bold, italic
            buf_text = text
        else:
            buf_text += text

    def walk(el):
        name = local(el.tag)

        if name == "r":
            for child in el:
                cname = local(child.tag)
                if cname == "fldChar":
                    ftype = child.get(qn("w:fldCharType"))
                    if ftype == "begin":
                        flush_buf()
                        fields.begin()
                    elif ftype == "separate":
                        fields.separate()
                    elif ftype == "end":
                        flush_buf()
                        out.append(fields.end())
                elif cname == "instrText":
                    fields.add_instr(child.text or "")
                elif cname in ("t", "tab", "br", "cr"):
                    if cname == "t":
                        text = child.text or ""
                    elif cname == "tab":
                        text = "\t"
                    else:
                        text = "\n"
                    if fields.active and fields.stack[-1]["mode"] == "instr":
                        continue
                    emit_plain(text, run_is_bold(el), run_is_italic(el))
            return

        if name == "hyperlink":
            flush_buf()
            rid = el.get(qn("r:id"))
            url = None
            if rid:
                try:
                    url = p.part.rels[rid].target_ref
                except Exception:
                    url = None
            saved = len(out)
            for child in el:
                walk(child)
            flush_buf()
            inner_text = "".join(out[saved:])
            del out[saved:]
            if url and not str(url).startswith("#"):
                out.append(f"[{inner_text.strip() or url}]({url})")
            else:
                out.append(inner_text)
            return

        if name in (
            "commentRangeStart",
            "commentRangeEnd",
            "bookmarkStart",
            "bookmarkEnd",
            "proofErr",
        ):
            return
        if name in ("drawing", "pict", "object"):
            return

        for child in el:
            walk(child)

    walk(p._element)
    flush_buf()
    return "".join(out).replace("\u200b", "").strip()


def unwrap_heading(text: str) -> str:
    """Heading styles are already bold in Word; drop wrapping ** / * on the whole title."""
    t = text.strip()
    for _ in range(3):
        m = re.fullmatch(r"\*\*\*(.+)\*\*\*", t, re.S)
        if m:
            t = m.group(1).strip()
            continue
        m = re.fullmatch(r"\*\*(.+)\*\*", t, re.S)
        if m:
            t = m.group(1).strip()
            continue
        m = re.fullmatch(r"\*(.+)\*", t, re.S)
        if m:
            t = m.group(1).strip()
            continue
        break
    return t


def heading_level(style_name: str | None) -> int | None:
    if not style_name:
        return None
    m = re.match(r"Heading\s+(\d+)$", style_name, re.I)
    return int(m.group(1)) if m else None


def is_toc_style(style_name: str | None) -> bool:
    return bool(style_name) and style_name.lower().startswith("toc")


def convert_list_prefix(text: str) -> str:
    if text.startswith("• "):
        return "- " + text[2:]
    if text.startswith("•"):
        return "- " + text[1:].lstrip()
    return text


def is_pre_line(text: str) -> bool:
    if not text:
        return False
    if text.startswith(("├", "│", "└", "─")):
        return True
    if re.match(r"^[A-Za-z0-9_.\-]+/\s*$", text):
        return True
    if re.match(r"^(dataset|data|meta|videos|episodes|images)/", text):
        return True
    if text.startswith("    ") or text.startswith("\t"):
        return True
    return False


def table_to_md(table: Table) -> str:
    rows_md = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            parts = []
            for p in cell.paragraphs:
                if is_drawing_only(p):
                    continue
                t = paragraph_to_md(p)
                if t:
                    parts.append(t)
            cell_text = "<br>".join(parts).replace("|", "\\|")
            cell_text = re.sub(r"\s*\n\s*", " ", cell_text).strip()
            cells.append(cell_text)
        rows_md.append(cells)

    if not rows_md:
        return ""

    ncols = max(len(r) for r in rows_md)
    for r in rows_md:
        while len(r) < ncols:
            r.append("")

    lines = [
        "| " + " | ".join(rows_md[0]) + " |",
        "| " + " | ".join(["---"] * ncols) + " |",
    ]
    for r in rows_md[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def iter_block_items(parent):
    body = parent.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


def postprocess_pre_blocks(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if is_pre_line(line):
            block = [line]
            i += 1
            while i < n:
                if lines[i] == "":
                    j = i + 1
                    while j < n and lines[j] == "":
                        j += 1
                    if j < n and is_pre_line(lines[j]):
                        i = j
                        block.append(lines[j])
                        i += 1
                        continue
                    break
                if is_pre_line(lines[i]):
                    block.append(lines[i])
                    i += 1
                    continue
                break
            if len(block) >= 2 and any(x.startswith(("├", "│", "└")) for x in block):
                out.append("```")
                out.extend(block)
                out.append("```")
            else:
                out.extend(block)
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def convert(docx_path: Path, md_path: Path) -> str:
    doc = Document(str(docx_path))
    lines: list[str] = []
    title_done = False
    prev_blank = True

    def emit(s: str = ""):
        nonlocal prev_blank
        if s == "":
            if not prev_blank:
                lines.append("")
                prev_blank = True
            return
        lines.append(s)
        prev_blank = False

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            style = block.style.name if block.style else "Normal"
            if is_toc_style(style):
                continue

            text_plain = (block.text or "").strip()
            if text_plain == "目录":
                continue

            hl = heading_level(style)
            if hl is not None:
                md_text = unwrap_heading(paragraph_to_md(block))
                md_text = re.sub(r"\t\d+$", "", md_text).strip()
                if not md_text:
                    continue

                if not title_done and hl == 1:
                    emit(f"# {md_text}")
                    title_done = True
                    emit("")
                    continue

                level = 2 if hl == 1 else hl + 1
                emit("")
                emit("#" * level + " " + md_text)
                emit("")
                continue

            if is_drawing_only(block):
                continue

            md = paragraph_to_md(block)
            if not md.strip():
                continue

            md = convert_list_prefix(md)

            if re.match(r"^(图：|示意：|【插图地址】)", md):
                if re.match(r"^(图：|示意：)", md):
                    emit(f"*{md}*")
                else:
                    emit(md)
                emit("")
                continue

            emit(md)
            emit("")

        elif isinstance(block, Table):
            emit(table_to_md(block))
            emit("")

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = postprocess_pre_blocks(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip() + "\n"
    md_path.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert DOCX reports to Markdown.")
    parser.add_argument("input", type=Path, help="Input .docx")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="Output .md (default: same stem next to the docx)",
    )
    args = parser.parse_args(argv)

    src = args.input.expanduser().resolve()
    if not src.is_file():
        print(f"error: not found: {src}", file=sys.stderr)
        return 1
    dst = args.output.expanduser().resolve() if args.output else src.with_suffix(".md")
    dst.parent.mkdir(parents=True, exist_ok=True)

    text = convert(src, dst)
    n_links = len(re.findall(r"\[([^\]]*)\]\((https?://[^)]+)\)", text))
    n_heads = sum(1 for line in text.splitlines() if line.startswith("#"))
    print(f"Wrote {dst} ({dst.stat().st_size} bytes, links={n_links}, headings={n_heads})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
