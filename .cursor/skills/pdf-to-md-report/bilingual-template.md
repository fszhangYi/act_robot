# Bilingual paragraph template（中英段落对照）

Use when the user asks for **中英对照** / bilingual MD after PDF→EN Markdown.

## Header

```markdown
# 中文标题
# English Title

> **文档说明：** 由 [`paper.pdf`](paper.pdf) 抽取并整理为 Markdown，再做成**段落级中英对照**。
> 图用 PDF 截图/内嵌图导出后插入（见 `assets/<stem>/`）；表格与公式已转写为 Markdown。
> 无法可靠转换处保留 `【图注占位】`。
> **arXiv:** [XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

**Authors / 作者：** …
```

## Body blocks

```markdown
## Abstract / 摘要

**EN:** …

**ZH:** …

## I. Introduction / 引言

**EN:** …

**ZH:** …
```

## Figures

```markdown
![Fig. 1 / 图 1](assets/<stem>/fig1.png)

**Fig. 1 / 图 1：** English caption. / 中文图注。
```

## References

Keep bibliographic lines in the **original language** unless the user explicitly wants each entry translated. State that policy once in the doc note.
