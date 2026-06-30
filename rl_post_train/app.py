#!/usr/bin/env python3
"""Small browser viewer for tau2 live conversations."""

from __future__ import annotations

import argparse
import functools
import json
import re
import socket
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parent
LIVE_DIR = ROOT / "external" / "tau2-bench" / "data" / "live_conversations"
SIM_DIR = ROOT / "external" / "tau2-bench" / "data" / "simulations"
LIVE_LATEST_RESULTS = LIVE_DIR / "latest_results.json"


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>tau2 Conversation Viewer</title>
  <style>
    :root {
      --bg: #f6f7f8;
      --panel: #fff;
      --line: #d7dde3;
      --text: #18222d;
      --muted: #627080;
      --agent: #174a75;
      --user: #17634f;
      --tool: #755018;
      --fail: #9d2d2d;
      --code: #101923;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    header {
      position: sticky;
      top: 0;
      z-index: 2;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 12px 18px;
    }

    .bar {
      max-width: 1180px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: auto auto 92px auto 1fr;
      align-items: center;
      gap: 10px;
    }

    h1 {
      margin: 0;
      font-size: 17px;
      letter-spacing: 0;
      white-space: nowrap;
    }

    input, button {
      font: inherit;
      border: 1px solid var(--line);
      border-radius: 6px;
      min-height: 36px;
      background: #fff;
      color: var(--text);
    }

    input {
      width: 100%;
      padding: 7px 9px;
    }

    button {
      padding: 7px 11px;
      cursor: pointer;
    }

    button:hover { background: #f0f3f6; }

    button:disabled {
      color: #a0a8b0;
      cursor: default;
      background: #f5f6f7;
    }

    .hint {
      color: var(--muted);
      font-size: 13px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 18px 20px 40px;
    }

    .summary {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 1px;
      border: 1px solid var(--line);
      background: var(--line);
      border-radius: 8px;
      overflow: hidden;
      margin-bottom: 22px;
    }

    .stat {
      min-width: 0;
      background: var(--panel);
      padding: 10px 12px;
    }

    .label {
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 4px;
    }

    .value {
      font-weight: 750;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .value.success { color: var(--user); }
    .value.failed { color: var(--fail); }
    .value.unknown { color: var(--muted); }

    .timeline {
      display: grid;
      gap: 18px;
    }

    .msg {
      display: grid;
      grid-template-columns: 86px minmax(0, 1fr);
      gap: 16px;
      align-items: start;
    }

    .side {
      position: sticky;
      top: 82px;
      display: grid;
      justify-items: end;
      gap: 5px;
      padding-top: 4px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.25;
    }

    .role {
      display: inline-block;
      border-radius: 999px;
      padding: 4px 8px;
      color: #fff;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.02em;
      font-size: 11px;
    }

    .assistant .role { background: var(--agent); }
    .user .role { background: var(--user); }
    .tool .role { background: var(--tool); }

    .step {
      white-space: nowrap;
    }

    .body {
      max-width: 900px;
      border-radius: 8px;
      padding: 18px 20px;
      background: var(--panel);
      box-shadow: 0 0 0 1px var(--line);
      font-size: 19px;
      line-height: 1.62;
      overflow-wrap: anywhere;
    }

    .body p { margin: 0 0 12px; }
    .body p:last-child { margin-bottom: 0; }

    .user .body {
      background: #f1faf6;
    }

    .assistant .body {
      background: #fff;
    }

    .tool .body {
      background: #fffaf1;
    }

    details {
      margin-bottom: 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfd;
    }

    summary {
      cursor: pointer;
      padding: 8px 10px;
      color: var(--muted);
      font-weight: 700;
      font-size: 13px;
    }

    .think {
      border-top: 1px solid var(--line);
      padding: 10px;
      white-space: pre-wrap;
      color: #394756;
      font-size: 15px;
      line-height: 1.5;
    }

    .tool-call {
      margin-top: 12px;
      border: 1px solid #dec99f;
      border-radius: 6px;
      overflow: hidden;
    }

    .tool-name {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 8px 10px;
      background: #fbf5ea;
      color: var(--tool);
      font-size: 13px;
      font-weight: 800;
    }

    pre {
      margin: 0;
      padding: 12px;
      overflow: auto;
      background: var(--code);
      color: #e8edf3;
      font: 14px/1.45 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }

    .empty {
      display: grid;
      place-items: center;
      min-height: 300px;
      color: var(--muted);
      text-align: center;
    }

    @media (max-width: 760px) {
      .bar { grid-template-columns: auto 86px auto; }
      h1, .hint { grid-column: 1 / -1; }
      .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .msg {
        grid-template-columns: 1fr;
        gap: 6px;
      }
      .side {
        position: static;
        justify-items: start;
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .body {
        max-width: none;
        font-size: 18px;
      }
    }
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <h1>tau2 Conversation Viewer</h1>
      <input id="index" type="text" inputmode="numeric" pattern="[0-9]*" value="0" aria-label="Conversation index">
      <button id="prev" type="button">Prev</button>
      <button id="next" type="button">Next</button>
      <div class="hint" id="hint">Loading conversations...</div>
    </div>
  </header>

  <main>
    <section class="summary" id="summary"></section>
    <section class="timeline" id="timeline">
      <div class="empty">Use Prev / Next, or type an index and press Enter.</div>
    </section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    const state = { items: [], current: null, index: 0 };

    function esc(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function fmtDate(value) {
      if (!value) return "-";
      const d = new Date(value);
      return Number.isNaN(d.getTime()) ? value : d.toLocaleString();
    }

    function fmtDuration(seconds) {
      if (seconds === null || seconds === undefined) return "-";
      if (seconds < 60) return `${seconds.toFixed(1)}s`;
      return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
    }

    function prettyJson(value) {
      if (value === null || value === undefined || value === "") return "";
      if (typeof value !== "string") return JSON.stringify(value, null, 2);
      try { return JSON.stringify(JSON.parse(value), null, 2); }
      catch { return value; }
    }

    function splitThink(content) {
      const text = String(content ?? "");
      const match = text.match(/<think>\s*([\s\S]*?)\s*<\/think>/i);
      if (!match) return { think: "", visible: text };
      return { think: match[1].trim(), visible: text.replace(match[0], "").trim() };
    }

    function paragraphs(text) {
      const clean = esc(text || "");
      if (!clean) return "";
      return clean.split(/\n{2,}/).map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`).join("");
    }

    async function api(path) {
      const response = await fetch(path);
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    async function init() {
      const data = await api("/api/list");
      state.items = data.conversations;
      $("hint").textContent = `${state.items.length} tasks available, index 0-${Math.max(0, state.items.length - 1)}`;
      if (state.items.length) loadIndex(0);
    }

    async function loadIndex(index) {
      const idx = Number.parseInt(index, 10);
      if (!Number.isInteger(idx) || idx < 0 || idx >= state.items.length) {
        $("timeline").innerHTML = `<div class="empty">Index must be between 0 and ${state.items.length - 1}.</div>`;
        return;
      }
      $("index").value = idx;
      state.index = idx;
      state.current = await api(`/api/conversation/${idx}`);
      render();
    }

    function renderSummary(summary) {
      const outcomeClass = (summary.outcome || "unknown").toLowerCase();
      const stats = [
        { label: "Task", value: summary.task_id ?? "-" },
        { label: "Outcome", value: summary.outcome_label ?? "Unknown", klass: outcomeClass },
        { label: "Messages", value: summary.message_count },
        { label: "Tools", value: summary.tool_call_count },
        { label: "Duration", value: fmtDuration(summary.duration_seconds) },
      ];
      $("summary").innerHTML = stats.map((item) => `
        <div class="stat">
          <div class="label">${esc(item.label)}</div>
          <div class="value ${esc(item.klass || "")}">${esc(item.value)}</div>
        </div>
      `).join("");
      $("hint").textContent = `Index ${summary.index}: ${summary.name} · result source: ${summary.outcome_source || "none"}`;
    }

    function renderMessage(row) {
      const msg = row.message || {};
      const role = msg.role || "unknown";
      const content = splitThink(msg.content);
      const toolCalls = Array.isArray(msg.tool_calls) ? msg.tool_calls : [];
      const roleLabel = role === "assistant" ? "agent" : role === "tool" ? "tool result" : role;

      const thinkHtml = content.think
        ? `<details><summary>Reasoning</summary><div class="think">${esc(content.think)}</div></details>`
        : "";
      const textHtml = role === "tool" ? "" : paragraphs(content.visible);
      const resultHtml = role === "tool" ? `<pre>${esc(prettyJson(msg.content))}</pre>` : "";
      const callsHtml = toolCalls.map((call) => `
        <div class="tool-call">
          <div class="tool-name">
            <span>${esc(call.name || "tool_call")}</span>
            <span>${esc(call.id || "")}</span>
          </div>
          <pre>${esc(prettyJson(call.arguments ?? call))}</pre>
        </div>
      `).join("");

      return `
        <article class="msg ${esc(role)}">
          <div class="side">
            <span class="role">${esc(roleLabel)}</span>
            <span class="step">step ${esc(row.step)}</span>
          </div>
          <div class="body">
            ${thinkHtml}
            ${textHtml}
            ${callsHtml}
            ${resultHtml}
          </div>
        </article>
      `;
    }

    function render() {
      if (!state.current) return;
      renderSummary(state.current.summary);
      $("prev").disabled = state.index <= 0;
      $("next").disabled = state.index >= state.items.length - 1;
      $("timeline").innerHTML = state.current.messages.map(renderMessage).join("");
    }

    $("prev").addEventListener("click", () => loadIndex(state.index - 1).catch(showError));
    $("next").addEventListener("click", () => loadIndex(state.index + 1).catch(showError));
    $("index").addEventListener("input", () => {
      $("index").value = $("index").value.replace(/[^0-9]/g, "");
    });
    $("index").addEventListener("keydown", (event) => {
      if (event.key === "Enter") loadIndex($("index").value).catch(showError);
    });

    function showError(error) {
      $("timeline").innerHTML = `<div class="empty">Error: ${esc(error.message || error)}</div>`;
    }

    init().catch(showError);
  </script>
</body>
</html>
"""


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return rows


def parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def elapsed(start: Any, end: Any) -> float | None:
    start_dt = parse_time(start)
    end_dt = parse_time(end)
    if not start_dt or not end_dt:
        return None
    return max(0.0, (end_dt - start_dt).total_seconds())


def task_number(path: Path) -> int:
    match = re.search(r"airline_task_(\d+)_", path.name)
    return int(match.group(1)) if match else 999999


def conversation_dirs() -> list[Path]:
    if not LIVE_DIR.exists():
        return []
    paths = [
        p for p in LIVE_DIR.iterdir()
        if p.is_dir() and (p / "conversation.md").exists()
    ]
    return sorted(paths, key=lambda p: (task_number(p), p.name))


def live_dirs_by_simulation_id(batch_id: str | None = None) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in conversation_dirs():
        metadata = read_json(path / "metadata.json")
        if batch_id is not None and metadata.get("batch_id") != batch_id:
            continue
        simulation_id = metadata.get("simulation_id")
        if simulation_id:
            paths[simulation_id] = path
    return paths


def live_dirs_by_task_id(batch_id: str | None = None) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in conversation_dirs():
        metadata = read_json(path / "metadata.json")
        if batch_id is not None and metadata.get("batch_id") != batch_id:
            continue
        task_id = metadata.get("task_id")
        if task_id is None:
            parsed = task_number(path)
            task_id = parsed if parsed != 999999 else None
        if task_id is not None:
            current = paths.get(str(task_id))
            if current is None or path.stat().st_mtime > current.stat().st_mtime:
                paths[str(task_id)] = path
    return paths


def result_candidates() -> list[Path]:
    if LIVE_LATEST_RESULTS.exists():
        return [LIVE_LATEST_RESULTS]
    if not SIM_DIR.exists():
        return []
    return sorted(
        SIM_DIR.rglob("results.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


@functools.cache
def latest_results() -> tuple[Path | None, list[dict[str, Any]], str | None]:
    for path in result_candidates():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        simulations = data.get("simulations") if isinstance(data, dict) else None
        if not isinstance(simulations, list):
            continue
        rows = [sim for sim in simulations if isinstance(sim, dict)]
        batch_id = data.get("_tau2_live_batch_id") if isinstance(data, dict) else None
        return path, rows, batch_id
    return None, [], None


def outcome_from_result(result: dict[str, Any] | None, fallback_reason: str = "") -> dict[str, str]:
    if result is None:
        return {
            "outcome": "unknown",
            "outcome_label": "Unknown",
            "outcome_source": fallback_reason or "No matching results.json found",
        }

    reward_info = result.get("reward_info") or {}
    reward = reward_info.get("reward")
    termination = result.get("termination_reason")
    failed_by_infra = termination in {
        "infrastructure_error",
        "agent_error",
        "environment_error",
    }
    success = reward == 1.0 and not failed_by_infra
    label = "Success" if success else "Failed"
    if isinstance(reward, (int, float)):
        label = f"{label} ({reward:g})"
    return {
        "outcome": "success" if success else "failed",
        "outcome_label": label,
        "outcome_source": "results.json",
    }


def result_sort_key(result: dict[str, Any]) -> tuple[int, str]:
    task_id = result.get("task_id")
    try:
        return int(task_id), str(result.get("id") or "")
    except (TypeError, ValueError):
        return 999999, str(result.get("id") or "")


def latest_batch_entries() -> list[dict[str, Any]]:
    results_path, results, batch_id = latest_results()
    if results:
        live_by_id = live_dirs_by_simulation_id(batch_id)
        live_by_task = live_dirs_by_task_id(batch_id) if batch_id is not None else {}
        entries = []
        for result in sorted(results, key=result_sort_key):
            live_path = live_by_id.get(result.get("id"))
            if (
                live_path is None
                and batch_id is not None
                and result.get("task_id") is not None
            ):
                live_path = live_by_task.get(str(result.get("task_id")))
            entries.append(
                {
                    "result": result,
                    "result_source": (
                        str(results_path.relative_to(ROOT))
                        if results_path
                        else "results.json"
                    ),
                    "path": live_path,
                }
            )
        return entries

    return [
        {"path": path, "result": None, "result_source": None}
        for path in conversation_dirs()
    ]


def summarize_entry(entry: dict[str, Any], index: int) -> dict[str, Any]:
    path = entry.get("path")
    result = entry.get("result")
    result_source = entry.get("result_source")
    metadata = read_json(path / "metadata.json") if path else {}
    messages = read_jsonl(path / "conversation.jsonl") if path else []
    events = read_jsonl(path / "events.jsonl") if path else []
    timestamps = [metadata.get("started_at")]
    timestamps += [
        m.get("timestamp") or m.get("message", {}).get("timestamp")
        for m in messages
    ]
    timestamps += [e.get("timestamp") for e in events]
    timestamps = [t for t in timestamps if t]
    started = metadata.get("started_at") or (min(timestamps) if timestamps else None)
    updated = max(timestamps) if timestamps else None
    tool_count = sum(
        len(m.get("message", {}).get("tool_calls") or [])
        for m in messages
        if isinstance(m.get("message"), dict)
    )
    task_id = metadata.get("task_id")
    simulation_id = metadata.get("simulation_id")
    if result:
        task_id = result.get("task_id")
        simulation_id = result.get("id")
    if task_id is None:
        task_id = str(task_number(path)) if path and task_number(path) != 999999 else "-"
    if result:
        outcome = outcome_from_result(result)
        outcome["outcome_source"] = result_source or "results.json"
    else:
        outcome = outcome_from_result(None, "No latest results file found")
    return {
        "index": index,
        "name": path.name if path else f"airline_task_{task_id}_{simulation_id}",
        "task_id": task_id,
        "simulation_id": simulation_id,
        **outcome,
        "started_at": started,
        "updated_at": updated,
        "duration_seconds": (
            result.get("duration")
            if result and result.get("duration") is not None
            else elapsed(started, updated)
        ),
        "message_count": len(messages),
        "tool_call_count": tool_count,
        "has_conversation": path is not None,
    }


def payload_for(index: int) -> dict[str, Any]:
    entries = latest_batch_entries()
    if index < 0 or index >= len(entries):
        raise IndexError(index)
    entry = entries[index]
    path = entry.get("path")
    messages = read_jsonl(path / "conversation.jsonl") if path else []
    events = read_jsonl(path / "events.jsonl") if path else []
    guardrail_messages = [
        {
            "step": event.get("step"),
            "timestamp": event.get("timestamp"),
            "message": {
                "role": "tool",
                "content": {
                    "guardrail": True,
                    **(event.get("details") or {}),
                },
            },
        }
        for event in events
        if event.get("event") == "guardrail_decision"
    ]
    if guardrail_messages:
        messages = sorted(
            [*messages, *guardrail_messages],
            key=lambda row: (row.get("timestamp") or "", str(row.get("step") or "")),
        )
    if not messages:
        summary = summarize_entry(entry, index)
        infra_events = [
            event for event in events
            if event.get("event") == "infrastructure_error"
        ]
        if infra_events:
            details = infra_events[-1].get("details") or {}
            content = "\n".join(
                part
                for part in [
                    "Infrastructure error recorded before a full transcript was available.",
                    details.get("error_type"),
                    details.get("error"),
                ]
                if part
            )
        else:
            content = (
                "No live conversation transcript was saved for this simulation. "
                f"Outcome is available from {summary.get('outcome_source')}."
            )
        messages = [
            {
                "step": 0,
                "timestamp": summary.get("started_at"),
                "message": {
                    "role": "tool",
                    "content": content,
                },
            }
        ]
    return {
        "summary": summarize_entry(entry, index),
        "metadata": read_json(path / "metadata.json") if path else {},
        "messages": messages,
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = unquote(self.path.split("?", 1)[0])
        if path == "/":
            return self.send_html(HTML)
        if path == "/api/list":
            entries = latest_batch_entries()
            return self.send_json({
                "conversations": [
                    summarize_entry(entry, i)
                    for i, entry in enumerate(entries)
                ]
            })
        if path.startswith("/api/conversation/"):
            try:
                index = int(path.rsplit("/", 1)[-1])
                return self.send_json(payload_for(index))
            except (ValueError, IndexError):
                return self.send_json({"error": "conversation not found"}, HTTPStatus.NOT_FOUND)
        if path == "/health":
            return self.send_json({"ok": True})
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.client_address[0]} - {fmt % args}")

    def send_html(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def open_port(host: str, preferred: int) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError("no open port found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    port = open_port(args.host, args.port)
    server = ThreadingHTTPServer((args.host, port), Handler)
    url = f"http://{args.host}:{port}"
    print(f"Open {url}", flush=True)
    print(f"Reading {LIVE_DIR}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
