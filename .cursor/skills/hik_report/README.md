# hik_report

报告**文风**、**内容总结**与**汇报向逐句注解** skill；不绑定单一技术主题。

## 安装

```bash
cp -a hik_report /path/to/workspace/.cursor/skills/hik_report
# 或
cp -a hik_report ~/.cursor/skills/hik_report
```

Cursor 只扫描 `.cursor/skills/` 与 `~/.cursor/skills/`。

## 打包

```bash
rm -f hik_report.zip
zip -r hik_report.zip hik_report -x '*.pyc' -x '*__pycache__*'
```

## 文件

| 文件 | 作用 |
| --- | --- |
| `SKILL.md` | 主流程：文风 / 总结 / 逐句详解 |
| `style-guide.md` | 句式与禁忌 |
| `annotation-template.md` | verbose 汇报稿模板 |
| `format-requirements.txt` | 导出 DOCX 时的排版硬规则（可选） |
