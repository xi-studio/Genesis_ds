"""
MolAgent Core (DeepSeek) — perceive / infer / host loop.

Python on the host runs only via the **exec** function tool (default tool bundle).

Modules:
    tools            — tool dispatch, OpenAI-style schemas, built-in handlers (exec, shell, grep, …)
    config           — built-in tool schemas; replace with your own non-empty ``tools`` list if needed
    output           — serialized prints and log file
    tokenizer        — CJK vs Latin token heuristic (auto-calibrated from API usage)
    consciousness    — ``agent_db_file``: consciousness_messages + agent_state (infer window)
    core_memory      — same ``agent_db_file``: core_memory_entries + core_memory_snapshot
    prompt           — single system prompt (tools + workspace + pacing)
    infer            — LLM chat + tool loop (streaming / non-streaming)
    exec_engine      — Python runner for the ``exec`` tool
    host_primitives  — trigger and signal handling
    loop_control     — tail tokens, chain/wait, gap timing
    ui_stub          — UI event broadcast stub (overridden by server)
"""
