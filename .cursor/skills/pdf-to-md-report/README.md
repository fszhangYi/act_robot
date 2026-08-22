# pdf-to-md-report

Cursor Agent Skill：将研究类 PDF（尤其 arXiv / 双栏论文）整理为 Markdown；导出插图、表格/公式转写，可选中英段落对照。

与 [`docx-to-md-report`](../docx-to-md-report/)（Word→MD）、[`md-to-docx-report`](../md-to-docx-report/)（MD→Word）互补。

## 安装位置

Cursor **只会**自动发现这两个位置下的 skill（目录内需有 `SKILL.md`）：

| 类型 | 路径 |
| --- | --- |
| 项目级 | `<工作区>/.cursor/skills/<skill-name>/` |
| 个人级 | `~/.cursor/skills/<skill-name>/` |

```bash
cp -a pdf-to-md-report /path/to/workspace/.cursor/skills/pdf-to-md-report
# 或
cp -a pdf-to-md-report ~/.cursor/skills/pdf-to-md-report
```

## 安装依赖

```bash
pip install pymupdf pillow
```

## 命令行

```bash
python scripts/pdf_to_md.py input.pdf output.md --extract-images --list-captions
python scripts/pdf_to_md.py docs/paper.pdf   # → docs/paper.md + docs/assets/paper/
```

裁剪指定区域（PDF 点坐标，页码 0-based）：

```bash
python scripts/pdf_to_md.py paper.pdf /tmp/out.md \
  --assets-dir docs/assets/paper \
  --crop '0,fig1,50,300,560,560'
```

## 文件

| 文件 | 作用 |
| --- | --- |
| `SKILL.md` | Agent 主流程与转换约定 |
| `bilingual-template.md` | 中英段落对照模板 |
| `scripts/pdf_to_md.py` | 文本抽取、内嵌图、图注检索、区域裁剪 |

## 打包

在仓库根目录：

```bash
rm -f pdf-to-md-report.zip
zip -r pdf-to-md-report.zip pdf-to-md-report -x '*.pyc' -x '*__pycache__*'
```
