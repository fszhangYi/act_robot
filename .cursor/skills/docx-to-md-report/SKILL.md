---
name: docx-to-md-report
description: >-
  Convert Chinese research Word reports (.docx) to Markdown with hyperlink
  restoration as [text](url), TOC skipped, binary images skipped, Heading
  round-trip aligned with md-to-docx-report. Use when the user asks to restore
  STAGE*.docx / plus*.docx / script_plus.docx / chapter*.docx to .md, docx→md,
  转成 md, or 超链接还原成 []().
---

# DOCX → Markdown（调研报告还原）

与 **md-to-docx-report** 配对：把 Word 定稿还原成可再编辑的 Markdown。

## When to use

- User asks to convert `*.docx` → `*.md` under a report project
- Mentions `转成 md` / `还原成 md` / `docx→md` / 超链接写成 `[]()`
- Restoring `STAGE*.docx`, `plus*.docx`, `script_plus.docx`, `chapter*.docx`

## Conversion rules

1. Output **same stem** `.md` next to the docx unless the user names another path
2. **Hyperlinks must become `[文字](url)`** — parse `w:hyperlink` and `HYPERLINK "url"` fields (instrText may be split across runs). Do not drop URLs
3. Internal TOC links (`HYPERLINK \l …`) stay as display text only
4. Skip **TOC styles** and the paragraph `目录`
5. **Do not export binary images**; skip drawing-only paragraphs. Keep text `【插图地址】…` and captions `图：` / `示意：`
6. Captions `图：…` / `示意：…` → italic `*图：…*`
7. GFM tables; `•` bullets → `- `
8. Directory trees (`├` / `│` / `└`) → fenced code blocks
9. Heading round-trip with md-to-docx-report:

| Word | Markdown |
| --- | --- |
| first Heading 1 (document title) | `#` |
| later Heading 1 (chapters) | `##` |
| Heading 2 | `###` |
| Heading 3 | `####` |
| Heading 4 | `#####` |

## How to convert

Prefer the bundled script (do not reinvent field parsing):

```bash
pip install python-docx -q
python scripts/docx_to_md.py /path/to/input.docx /path/to/output.md
```

Same-name output:

```bash
python scripts/docx_to_md.py STAGE3.docx
# → STAGE3.md
```

## Agent checklist

1. Confirm input `.docx` and output `.md` path
2. Run `scripts/docx_to_md.py`
3. Spot-check: sample `[text](url)`, headings, one table, captions; images should be absent as binaries
4. Dependencies: `python-docx` only
5. For the reverse direction (md→docx typography), call **md-to-docx-report**
