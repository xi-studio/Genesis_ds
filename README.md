# Genesis Agent (DeepSeek)

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

启动后浏览器打开 `http://127.0.0.1:xxx`(自己配置的web_listen端口) 查看运行状态。

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