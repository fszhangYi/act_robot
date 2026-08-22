---
name: hik_report
description: >-
  Guide Chinese technical/research report writing style, content summarization,
  and briefing-ready sentence-level annotations (汇报详解). Use when the user asks
  for 文章风格, 内容总结, 行文思路, 逐句注解, verbose 汇报稿, chapter digests, or
  report rewrite/restructure without locking to one product topic. Optional
  DOCX/MD export follows docx 格式要求 via md-to-docx-report.
---

# hik_report — 报告文风、总结与汇报注解

本 skill 管三件事：**怎么写**、**怎么收束成总结**、**怎么做成可汇报的逐句详解**。  
不默认绑定某一产品线或技术专题；具体主题以用户当前文稿为准。

## When to use

- 用户要调整/提炼**文章风格**、**行文思路**、章节逻辑
- 用户要**内容总结**（全书摘要、章摘要、要点卡片）
- 用户要**汇报版详解** / 逐句注解 / `verbose.md` 类交付
- 用户要在改结构时保持论证节奏，而不是只改标题编号
- 需要顺带导出 DOCX 时，再调用 `md-to-docx-report`（排版规则见 [format-requirements.txt](format-requirements.txt)）

## Not this skill’s job

- 不把报告锁死在单一主题模板（例如固定只写某一厂商或某一模型族）
- 不替代领域事实核查：缺公开依据时写「边界/未知」，不编造指标
- 不重写 DOCX 排版引擎：导出走 `md-to-docx-report`

---

## 1. 文章风格（Style）

写/改中文技术调研或工程论证文时，优先对齐下列文风（与仓库长文实践一致，可迁移到任意主题）：

| 维度 | 做法 |
| --- | --- |
| 先框架后细节 | 章首用 2–4 句钉住本章问题与后文用法；表/列表承担对比，正文少堆定义 |
| 限定句 | 关键结论带边界（适用场景、阶段、不可外推范围），避免终局口号 |
| 证据句 | 可核验论断配 `[锚文字](url)` 或明确「内部材料 / 公开不足」 |
| 节奏 | 问题 → 分析框架 → 证据/案例 → 阶段判断 → 收束到下一章问题 |
| 称名 | 术语首次出现给中文+必要英文；后文统一，不玩新词 |
| 图表 | 表对齐「维度 × 对象」；图注说明「看什么 + 图源」；流程用可阅读文本框即可 |

改写时：**先动逻辑与限定，再动修辞**；不要为了文采削弱可验收性。

更多句式与禁忌见 [style-guide.md](style-guide.md)。

---

## 2. 内容总结（Summarize）

按用户要的粒度选一种，默认先问清受众（领导快读 / 同级评审 / 对外公开）：

| 模式 | 产出 | 规则 |
| --- | --- | --- |
| 一书一页 | 问题、主张、证据链、缺口、下一步 | 不出现未在原文出现的新结论 |
| 章摘要 | 每章 5–8 条要点 + 该章在全书中的功能 | 标明「承上 / 启下」 |
| 对照卡 | 用表收束多对象差异 | 列名稳定，空单元格写「原文未给」 |
| 风险/边界清单 | 原文明确的限制与未证明项 | 与「已证明项」分列 |

总结必须**可回溯**：每条要点能指到原文章节或原句；禁止把推测写成原文结论。

---

## 3. 汇报版详解（Verbose annotation）

用户要「每一句都注解、兼顾行文思路」时，按 [annotation-template.md](annotation-template.md) 输出 Markdown：

1. **章/节导读**：这段在全书论证链上的位置（为何此时出现）
2. **原句**：保持原文，一句一条（按中文句号/问号/叹号切分；分号处是否切开随语义判断）
3. **注解**：解释术语、隐含前提、与前后句关系；需要时点明「证据 / 限定 / 过渡 / 收束」角色
4. **节末一行**：本小节推进了哪一个问题

要求：

- 注解服务汇报口播与答问，不另起一篇平行论文
- 外链在注解中可复述用途（「此句用公开报道支撑××」）
- 表格：先总结表意，再按行或按关键单元格注解，避免只贴表不解释

---

## 4. 与排版工具的关系

当用户要 `.docx` 定稿时：

```bash
pip install python-docx -q
python md-to-docx-report/scripts/md_to_docx.py INPUT.md OUTPUT.docx
```

格式硬规则以 [format-requirements.txt](format-requirements.txt) 为准（与仓库 `docx 格式要求.txt` 同步）。  
docx→md 还原时：**超链接必须保留为 `[]()`**。

---

## Agent checklist

- [ ] 已确认任务是「文风 / 总结 / 逐句汇报详解」中的哪一种（或组合）
- [ ] 未把某一主题模板强加到无关文稿上
- [ ] 总结与注解均可回溯到原文
- [ ] 限定与证据分开写，不把愿景写成现状
- [ ] 若导出 DOCX，走 `md-to-docx-report` 并提示更新目录域
