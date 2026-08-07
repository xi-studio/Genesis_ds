# tokenizer_data

本地 tokenizer 文件，供 `agent/tokenizer.py` 做**精确 token 计数**。

## 文件

| 文件 | 来源 (HuggingFace) | 词表大小 | 用于模型 |
|------|-------------------|---------|---------|
| `deepseek.json` | `deepseek-ai/DeepSeek-V3` 的 `tokenizer.json` | 128,815 | **所有模型**（统一近似） |

## 工作方式

`agent/tokenizer.py` **统一使用 DeepSeek 词典**近似计数，不再按模型名选择。
- 词典可用 → 用 HF `tokenizers` 库逐 token 精确计数
- 文件缺失 / 库不可用 → 回退 CJK/Latin 启发式（不会崩溃）

## 为什么用单一词典

对真实混合语料（意识日志 / 代码 / 中文）的实测：GLM 词典与 DeepSeek 词典在
**聚合计数上仅差 ~0.4%**（单条消息波动 ±18%，求和后抵消）。上下文裁剪只看聚合值，
所以单词典足够；同时省去 19MB 的 `glm.json`，且对代码类内容 DeepSeek 偏保守（安全方向）。

## 更新 / 替换词典

1. 从对应 HF 仓库下载 `tokenizer.json`：
   ```python
   from huggingface_hub import hf_hub_download
   import shutil
   p = hf_hub_download(repo_id="deepseek-ai/DeepSeek-V3", filename="tokenizer.json")
   shutil.copy(p, "agent/tokenizer_data/deepseek.json")
   ```

## 依赖

`tokenizers>=0.20`（见项目 requirements.txt）。注意：**不需要** `transformers`。
