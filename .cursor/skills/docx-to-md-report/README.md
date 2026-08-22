# docx-to-md-report

Cursor Agent Skill：将调研报告 DOCX 还原为 Markdown（超链接写成 `[文字](url)`，不导出二进制图）。

与 [`md-to-docx-report`](../md-to-docx-report/) 方向相反、标题层级对齐，便于往返编辑。

## 安装位置

Cursor **只会**自动发现这两个位置下的 skill（目录内需有 `SKILL.md`）：

| 类型 | 路径 |
| --- | --- |
| 项目级 | `<工作区>/.cursor/skills/<skill-name>/` |
| 个人级 | `~/.cursor/skills/<skill-name>/` |

```bash
cp -a docx-to-md-report /path/to/workspace/.cursor/skills/docx-to-md-report
# 或
cp -a docx-to-md-report ~/.cursor/skills/docx-to-md-report
```

## 安装依赖

```bash
pip install python-docx
```

## 命令行

```bash
python scripts/docx_to_md.py input.docx output.md
python scripts/docx_to_md.py STAGE3.docx   # → STAGE3.md
```

## 打包

```bash
rm -f docx-to-md-report.zip
zip -r docx-to-md-report.zip docx-to-md-report -x '*.pyc' -x '*__pycache__*'
```
