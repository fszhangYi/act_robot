---
name: pdf-to-md-report
description: >-
  Convert research/arXiv PDFs to Markdown: extract text with PyMuPDF, rebuild
  tables and equations, export figures (embedded images or caption-guided crop),
  use text placeholders when conversion fails, optional EN/ZH paragraph bilingual
  MD. Use when the user asks pdf→md, PDF 转 Markdown, arXiv 论文转 md, 中英对照,
  图注占位, or to insert cropped figures from a PDF into Markdown.
---

# PDF → Markdown（论文 / 技术报告）

把 **PDF（尤其 arXiv / IEEE 双栏论文）** 整理成可读 Markdown；与 **docx-to-md-report** / **md-to-docx-report** 互补（Word 往返另走那两个 skill）。

## When to use

- User asks `pdf→md` / `PDF 转 md` / `转成 Markdown` / arXiv 论文可读化
- Need **中英段落对照**（`**EN:**` / `**ZH:**`）
- Figures cannot be vectorized → crop/export PNG + caption, or `【图注占位】`
- Mentions figure assets under `docs/assets/...` next to the MD

## Non-goals

- Do **not** use full-page PNG as the body of the MD (双栏会糊成一团)
- Do **not** claim bit-exact layout fidelity; goal is readable structure
- Scanned-only PDFs need OCR first (out of scope unless user asks)

## Conversion policy

| Content | Action |
| --- | --- |
| Title / authors / abstract / sections | Rebuild as Markdown headings + paragraphs |
| Tables | GFM tables (retype from extracted text; fix broken line wraps) |
| Equations | Prefer `\(…\)` / `\[…\]` or `$…$` / `$$…$$` LaTeX; keep numbering like `(1)` |
| Figures / architecture / plots | Prefer **embedded image extract** or **caption-guided crop** → `![…](assets/…/figN.png)` + caption |
| Unconvertible visuals | `> **【图注占位】**` + original caption + page hint |
| References | Keep bibliographic entries in **original language** unless user asks for translation |
| Hyperlinks | `[text](url)` when URLs are known (arXiv abs/pdf) |

## Recommended pipeline

Copy and track:

```
Task Progress:
- [ ] 1. Locate PDF; note page count / arXiv id
- [ ] 2. Run scripts/pdf_to_md.py (text + embeds + caption hits)
- [ ] 3. Structure EN Markdown (headings, tables, equations)
- [ ] 4. Export / crop figures; replace placeholders with ![](…)
- [ ] 5. (Optional) Build bilingual EN/ZH paragraph MD
- [ ] 6. Spot-check paths, captions, one table, one equation
```

### Step 1–2: Extract

Prefer the bundled script (do not reinvent PyMuPDF boilerplate):

```bash
pip install pymupdf pillow -q
python scripts/pdf_to_md.py /path/to/paper.pdf /path/to/paper.md \
  --assets-dir /path/to/assets/paper_stem \
  --extract-images --list-captions
```

Useful flags:

- `--render-pages` — write `page_XX.png` for **inspection only** (not for MD body)
- `--dpi 216` — sharper crops (~3× at 72 dpi base); default 216
- `--list-captions` — print `Fig.` / `Figure` / `TABLE` hit boxes to guide clips

Agent may also open the PDF with `pymupdf` / `fitz` directly when refining clip rectangles.

### Step 3: Structure English MD

1. Header: title, authors, arXiv link, path to source PDF
2. Sections `## I. …` matching paper (or plain `##` if not numbered)
3. Rebuild tables carefully (PDF text often splits cells across lines)
4. Keep math as LaTeX; do not leave garbled Unicode fractions as-is when the intent is clear

### Step 4: Figures (hard-won rules)

1. **Prefer embedded images** when the figure is a single clean xref (photos, some schematics)
2. Else **render a clip** around the figure: find caption with `page.search_for("Fig. N")`, crop **above** the caption (IEEE: caption below figure)
3. **Tighten for two-column**: coarse clips pull in the other column — use caption `x` to stay in-column, or prefer embeds
4. Architecture / plots: 2–3× render; large photos: resize so the MD repo stays small
5. Paths relative to the MD file, e.g. `assets/sam2grasp/fig1.png` when MD lives in `docs/`
6. Replace `【图注占位】` with `![Fig. N](assets/.../figN.png)` + bold caption line

Optional crop helper after you know rects (PDF points):

```bash
python scripts/pdf_to_md.py paper.pdf /tmp/unused.md \
  --assets-dir docs/assets/paper --crop '0,fig1,50,300,560,560' --crop '3,fig2,40,35,575,200'
```

Format: `page_index,name,x0,y0,x1,y1` (0-based page).

### Step 5: Optional bilingual MD

When the user asks 中英对照 / bilingual:

1. Keep structure parallel to the English MD
2. Each prose block: `**EN:** …` then `**ZH:** …` (paragraph-level, not sentence salad)
3. Doc note at top: source PDF, placeholder policy, arXiv id
4. Figure captions can be bilingual on one line: `**Fig. N / 图 N：** EN. / ZH.`
5. References: usually leave English; say so in the doc note

See [bilingual-template.md](bilingual-template.md).

### Step 6: Spot-check

- [ ] Images resolve from the MD’s directory
- [ ] No full-page dumps used as body
- [ ] Tables readable; equation (1) etc. present
- [ ] Placeholders only where still needed
- [ ] If user also wants DOCX → call **md-to-docx-report** on the MD

## How to convert (CLI)

```bash
pip install pymupdf pillow -q
python scripts/pdf_to_md.py input.pdf output.md --extract-images --list-captions
```

Same-stem default:

```bash
python scripts/pdf_to_md.py docs/paper.pdf
# → docs/paper.md  and  docs/assets/paper/…
```

## Agent checklist

1. Confirm PDF path and desired `.md` / assets directory
2. Run `scripts/pdf_to_md.py` then **edit** the draft into structured MD (script is a scaffold, not a finished paper)
3. Export figures; prefer embeds / tight crops over placeholders
4. Optional bilingual pass
5. Dependencies: `pymupdf` (required), `pillow` (optional resize)
6. Downstream Word export → **md-to-docx-report**
