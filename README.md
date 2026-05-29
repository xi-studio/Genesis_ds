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
```

或直接：

```bash
python main.py
```

启动后浏览器打开 `http://127.0.0.1:7003` 查看运行状态。

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
├── web/                 # Web 监控界面
└── workspace/           # Agent 工作目录（自动创建）
```
