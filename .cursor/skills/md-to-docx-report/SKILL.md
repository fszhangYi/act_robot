---
name: md-to-docx-report
description: >-
  Convert Chinese research Markdown reports to DOCX with fixed typography
  (宋体/Times New Roman, 小四 body, 五号 tables/code, narrow margins, 1.5 line
  spacing, centered page numbers, numbered assets/N.filename address line
  plus embedded scaled images, Word hyperlinks in RGB(0,102,204), 「」→“”,
  LaTeX→Word OMML, black/white code blocks, Mermaid flowchart→PNG (§19),
  TOC field). Use when the user asks to turn .md into .docx, regenerate
  plus*.docx / STAGE*.docx / script_plus.docx / chapter*.docx, apply
  docx 格式要求, or insert Mermaid flowcharts as PNG figures.
---

# Markdown → 调研报告 DOCX

## When to use

- User asks to convert `*.md` → `*.docx` under a report project
- User mentions `docx 格式要求` / 页码 / 目录 / 插图用地址代替 / 超链接颜色 / 插入图片 / 公式 / 代码块 / mermaid 流程图
- Regenerating `new_script.docx`, `plus9.docx`, `STAGE*.docx`, `script_plus.docx`, `chapter*.docx`, etc.

## Format rules (source of truth)

See [format-requirements.txt](format-requirements.txt)（与仓库根目录 `docx 格式要求.txt` 对齐；版式参照 `STAGE3.docx`）：

1. 中文宋体 / 英文 Times New Roman
2. 正文小四
3. 表格、代码五号
4. 页边距窄
5. 表格适应窗口
6. 行距 1.5
7. 页码底部居中，五号字体
8. 插图用地址代替（仍输出居中的 `【插图地址】assets/N.filename`）
9. 自动生成目录（Word TOC 域，级别 1–3）
10. 标题使用 Heading 样式以便显示在目录上
11. 表格和图片前后各 0.5 行间隔（脚本按小四×1.5 行距的一半，即 9pt 空白实现）
12. 图及图名称居中
13. Markdown `[文字](url)` 转为 Word 真超链接（可见文字可点，不是裸 URL 或 `[]()` 原文）
14. 插入超链接的文字颜色为 RGB(0,102,204)，字体中文为宋体、英文为 Times New Roman，并且字号与插入链接之前所在上下文一致（正文小四 / 表格五号 / 标题随 Heading 字号）
15. 插图地址按出现顺序编号为 `assets/N.原文件名`（docx 显示路径）；路径一律写成同级 `assets/` 下；**在写地址的同时尝试把图片嵌入 docx，并缩放到合适宽度**（默认约 14cm，过高则按高度再压）；找不到文件时仅保留地址行并警告
16. 强调引号规范化：正文中的「…」转为中文弯引号 “… ”
17. 公式转为 Word 公式：`\(…\)` / `$…$` / `\[…\]` / `$$…$$`，以及文中的 LaTeX 片段（如 `(s_t,a^*_{t,1})`、`a^*_{t,1}\neq a^*_{t,2}`）写成 OMML，不要当纯斜体文本
18. 围栏代码块：整段段落底纹黑色，代码文字白色、Consolas、五号
19. **流程图**：正文中适合用流程图表达的内容，先用 Mermaid 生成图（避免长宽比过大或过小），再转成 PNG，按 §15 放入同级 `assets/` 并在 Markdown 中引用后由 `md_to_docx.py` 嵌入

## Mermaid → PNG (§19)

在写/改 MD、且内容适合流程图时，**先出图再转 DOCX**：

1. 编写 Mermaid（`flowchart` / `sequenceDiagram` 等）；布局优先左右或适度上下展开，避免极端细长或扁宽
2. 渲染为 PNG（任选其一）：
   - 推荐：`python scripts/mermaid_to_png.py diagram.mmd assets/name.png`（经 mermaid.ink；需 Pillow）
   - 或本机 `@mermaid-js/mermaid-cli`（`mmdc`）等等价工具
3. 在 MD 中用居中 HTML 或等价写法引用，例如：

```html
<p align="center">
  <img src="assets/name.png" alt="流程说明" width="85%"/>
</p>
<p align="center"><em>图：……（示意）。</em></p>
```

4. 再运行 `md_to_docx.py`；脚本会按 §15 编号地址并嵌入缩放图

源 `.mmd` 可另存 `assets/mermaid_src/` 便于修订，**不必**把 Mermaid 源码贴进 Word。

## Reference document (STAGE3.docx)

`STAGE3.docx` 是当前版式样例，转换结果应对齐其约定：

- 标题：`Heading 1`–`Heading 4` 均可能出现；目录 TOC 只收 1–3 级（与样例一致）
- 图注：`图：…` / `示意：…` 居中、五号斜体；插图先写居中的 `【插图地址】assets/N.filename`，再嵌入缩放后的图片
- 超链接：字段/关系型真链接；显示文字中文宋体、英文 Times New Roman；颜色 **RGB(0,102,204)**（十六进制 `0066CC`）；字号继承上下文；样例中链接无强制下划线
- 页边距窄（约 1.27cm）、正文 1.5 倍行距、页脚页码居中五号

## How to convert

Prefer the bundled script (do not reinvent formatting):

```bash
pip install python-docx Pillow -q
python scripts/md_to_docx.py /path/to/input.md /path/to/output.docx
```

Mermaid 预渲染：

```bash
python scripts/mermaid_to_png.py assets/mermaid_src/flow.mmd assets/flow.png
```

Example from `report/`:

```bash
python md-to-docx-report/scripts/md_to_docx.py STAGE2.md script_plus.docx
```

## Agent checklist

1. Confirm input Markdown path and desired output `.docx` name
2. If prose fits a flowchart: draft Mermaid with balanced aspect ratio → PNG under `assets/` (§19) → reference in MD
3. Run `scripts/md_to_docx.py` (working directory can be repo or absolute paths)
4. Tell user: open in Word/WPS → update TOC field if prompted
5. MD `![](assets/...)` / `<img src="assets/...">` → 居中 `【插图地址】assets/N.filename` **并尝试嵌入**缩放图片 + 居中图注
6. `[text](url)` must become Word hyperlinks in RGB(0,102,204) with 宋体/Times New Roman and context font size
7. Normalize 「」 → “” in body text (§16)
8. LaTeX math → Word equations (§17); do not leave `a^*_{t,1}` as plain italic
9. Fenced code blocks: black background, white text (§18)
10. Dependencies: `python-docx`；嵌入图缩放与 `mermaid_to_png.py` 需要 Pillow；Mermaid 渲染默认走 mermaid.ink（需网络）

## Markdown expectations

- `#` document title → Heading 1 (centered) + TOC page break after title
- `##` → Heading 1（章节，与样例 STAGE3 一致）
- `###` → Heading 2；`####` → Heading 3；`#####` → Heading 4（深小节，如 `7.2.1.1`）
- GFM tables, fenced code, `-` / `1.` lists, `**bold**`, `` `code` ``, `[text](url)` supported
- HTML centered figures with `<img src="...">` and optional `<em>` caption supported
- `*图：...*` / `*示意：...*` captions are centered
- Flowcharts: pre-render Mermaid → PNG in `assets/`, do not paste raw Mermaid into the DOCX body
