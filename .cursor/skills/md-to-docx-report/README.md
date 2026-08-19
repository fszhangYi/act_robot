# md-to-docx-report

Cursor Agent Skill：按固定中文报告版式将 Markdown 转为 DOCX。

## 为什么在 Agent Skills 列表里找不到？

Cursor **只会**自动发现这两个位置下的 skill（目录内需有 `SKILL.md`）：

| 类型 | 路径 |
| --- | --- |
| 项目级 | `<工作区>/.cursor/skills/<skill-name>/` |
| 个人级 | `~/.cursor/skills/<skill-name>/` |

版式规则见 `format-requirements.txt`（与 `docx 格式要求.txt` 同步）。样例文档：仓库内 `STAGE3.docx`（标题层级、居中图注、超链接 RGB(0,102,204) 等）。

插图：docx 中先写 `【插图地址】assets/N.filename`，再尝试嵌入并缩放图片（§15）；强调引号「」自动转为 “”（§16）。

仅放在 `report/md-to-docx-report/` 或打成 zip **不会**出现在 Skills 面板。本仓库已同步安装到：

- `report/.cursor/skills/md-to-docx-report/`
- `~/.cursor/skills/md-to-docx-report/`

若仍看不到：重开当前工作区，或在 Cursor 设置里刷新 Skills。

## 安装依赖

```bash
pip install python-docx
```

## 命令行

```bash
python scripts/md_to_docx.py input.md output.docx
```

## 手动安装

```bash
cp -a md-to-docx-report /path/to/workspace/.cursor/skills/md-to-docx-report
# 或
cp -a md-to-docx-report ~/.cursor/skills/md-to-docx-report
```

## 打包

在仓库根目录：

```bash
rm -f md-to-docx-report.zip
zip -r md-to-docx-report.zip md-to-docx-report -x '*.pyc' -x '*__pycache__*'
```
