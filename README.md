# Genesis Agent (DeepSeek)

极简 Python 版 Genesis Agent，源自 [infero-net/infero](https://github.com/infero-net/infero)。

- **核心约 200 行**：推理-执行主循环极简实现，易读易改
- **DeepSeek-v4-pro 适配**：当前版本Agent完整针对deepseek开发适配
- **高效记忆系统**：极简上下文记忆，**缓存率 99%+**，长时运行仍保持低 token 开销。可持续长时运行不崩溃，保持agent能力。
- **极低成本**：**1 亿 tokens 消费不到 10 元**
- **强扩展性**：Tool、Skills、记忆等均可由 Agent **运行后添加**，比如自进化能力，可以Agent运行后扩展

---

## 安装并启动

需要 **Python 3.10+**。

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 API key
cp config.json.example config.json
# 编辑 config.json：填入 api_key

# 启动
bash run.sh

或直接

python main.py
```

启动后可在浏览器打开 `http://127.0.0.1:xxx`（`web_listen` 端口）查看运行状态。

- **无浏览器模式**：不打开浏览器也可以直接使用，Agent 会在 server 端自主运行，自动查看文件，无需交互。

---

## 目录结构

```
DB_ds/
├── main.py              # Agent 入口
├── core_loop.py         # 推理-执行主循环
├── run.sh               # 一键启动脚本
├── requirements.txt     # Python 依赖
├── config.json          # LLM 配置（需自行填入 API key）
├── config.json.example  # 配置模板
├── agent/               # Agent 框架
│   ├── infer.py         # LLM 推理
│   ├── config.py        # 配置加载
│   ├── consciousness.py # 意识日志
│   ├── core_memory.py   # 核心记忆
│   ├── exec_engine.py   # 工具执行
│   └── tools/           # 工具定义
├── web/                 # Web 监控交互界面
└── workspace/           # Agent 自有工作目录（由Agent自动创建、维护）
```

---

## 交互技巧

- **暂停运行**：在 Web 界面点击 `Stop`，或在 server 端按 `Ctrl+C`。
- **清零 Agent**：手动删除 `workspace/` 目录，Agent 的记忆、文档等状态会一并清空（下次启动会重新创建该目录）。
- **项目任务管理**：工作时叫Agent在其他你指定的目录新建、做你自己的任务项目。这样可以多个Agent同时使用，Agent重启删除也不影响，可清零Agent后给它目录地址，继续接力进行任务。