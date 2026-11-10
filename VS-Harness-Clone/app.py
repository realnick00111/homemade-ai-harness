from __future__ import annotations

import argparse
import ast
import difflib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import ollama


DEFAULT_MODEL = "qwen3.5:9b"
MAX_ACTIONS = 5
MAX_RESPONSE_RETRIES = 2
MAX_TURNS = 20
MAX_NO_ACTION_TURNS = 3
MAX_IDENTICAL_FAILURES = 2
MAX_TASK_HOURS = 8
OLLAMA_REQUEST_TIMEOUT = 60
MAX_FILE_BYTES = 100_000
MAX_RESPONSE_BYTES = 2_000_000
IGNORED_DIRECTORIES = {
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules",
    "__pycache__", ".pytest_cache", "dist", "build",
}
SENSITIVE_PARTS = {"secrets", "credentials"}
SYSTEM_PROMPT = """You are an AI coding assistant working only in the user's selected project.
Answer with one JSON object matching the supplied schema. Give brief, useful progress
updates in "message"; never reveal private chain-of-thought or hidden reasoning.
Inspect the project before editing. Read an existing file before you edit it. Use the
available tools for all project inspection and changes; never claim an edit, test, or
other action you did not actually perform. When a tool reports an error, explain it or
correct it in a later step. For example, create a file with
{"type":"write_file","path":"example.py","content":"print('hello')\\n"}.
Read an existing file with {"type":"read_file","path":"example.py"}. Include every
required path, query, and content argument in each action. Run the tests after a
relevant code change. Set done=true only after the requested work and verification
have actually succeeded. Listing files or announcing a plan is not task completion;
when asked to inspect, read relevant project files, and when asked to implement,
make a successful write_file action before finishing.
"""
ACTION_TYPES = (
    "list_files", "search_files", "search_text", "read_file",
    "write_file", "check_python", "run_tests",
)
RESPONSE_FORMAT = {
    "type": "object",
    "properties": {
        "message": {"type": "string"},
        "done": {"type": "boolean"},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string", "enum": list(ACTION_TYPES),
                        "description": "Tool to actually execute in the project.",
                    },
                    "query": {"type": "string", "description": "Required for search_files and search_text."},
                    "path": {"type": "string", "description": "Project-relative file path for read_file, write_file, and check_python."},
                    "content": {"type": "string", "description": "Required complete file contents for write_file."},
                },
                "required": ["type"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["message", "done", "actions"],
    "additionalProperties": False,
}


def is_link_or_junction(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or (callable(is_junction) and is_junction())


def ensure_workspace(root: Path) -> Path:
    project_root = root.resolve()
    if not project_root.is_dir():
        raise ValueError(f"Project directory does not exist: {project_root}")

    workspace = project_root / "workspace"
    if is_link_or_junction(workspace):
        raise ValueError("The workspace folder cannot be a symbolic link or junction.")
    workspace.mkdir(exist_ok=True)
    if not workspace.is_dir():
        raise ValueError("The workspace path must be a directory.")

    resolved_workspace = workspace.resolve()
    try:
        resolved_workspace.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("The workspace must stay inside the project directory.") from exc
    if resolved_workspace != workspace:
        raise ValueError("The workspace folder cannot redirect to another directory.")
    return resolved_workspace


class Harness:
    def __init__(self, root: Path, max_task_seconds: float = MAX_TASK_HOURS * 3600) -> None:
        self.root = root.resolve()
        self.sessions: dict[str, list[dict[str, str]]] = {}
        self.snapshots: dict[str, dict[str, str]] = {}
        self.lock = threading.Lock()
        self.max_task_seconds = max_task_seconds
        self.client = ollama.Client(timeout=OLLAMA_REQUEST_TIMEOUT)
        self.active_tasks: dict[str, threading.Event] = {}

    def register_task(self, session_id: str) -> threading.Event:
        with self.lock:
            if session_id in self.active_tasks:
                raise ValueError("A task is already running in this chat.")
            cancel_event = threading.Event()
            self.active_tasks[session_id] = cancel_event
            return cancel_event

    def cancel_task(self, session_id: str) -> bool:
        with self.lock:
            cancel_event = self.active_tasks.get(session_id)
        if cancel_event is None:
            return False
        cancel_event.set()
        return True

    def finish_task(self, session_id: str, cancel_event: threading.Event) -> None:
        with self.lock:
            if self.active_tasks.get(session_id) is cancel_event:
                del self.active_tasks[session_id]

    def project_path(self, value: str) -> Path:
        if not value or Path(value).is_absolute():
            raise ValueError("Provide a non-empty project-relative path.")
        if any(part.lower() in SENSITIVE_PARTS or part.lower().startswith(".env")
               or part.lower().endswith((".pem", ".key", ".p12", ".pfx"))
               for part in Path(value).parts):
            raise ValueError("Reading or writing secrets and credentials is not allowed.")
        path = self.root
        for part in Path(value).parts:
            path = path / part
            if is_link_or_junction(path):
                raise ValueError("Paths through symbolic links or junctions are not allowed.")
        path = path.resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("The path must stay inside the project directory.") from exc
        return path

    def files(self) -> list[dict[str, Any]]:
        results = []
        for current, directories, filenames in os.walk(self.root, followlinks=False):
            base = Path(current)
            directories[:] = sorted(
                name for name in directories
                if name not in IGNORED_DIRECTORIES and not is_link_or_junction(base / name)
            )
            for name in sorted(filenames):
                path = base / name
                relative = path.relative_to(self.root)
                if any(part.lower() in SENSITIVE_PARTS or part.lower().startswith(".env")
                       or part.lower().endswith((".pem", ".key", ".p12", ".pfx"))
                       for part in relative.parts) or is_link_or_junction(path):
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                results.append({"path": relative.as_posix(), "size": size})
                if len(results) >= 500:
                    return results
        return results

    def action(self, action: dict[str, Any], session_id: str) -> tuple[str, dict[str, Any] | None]:
        kind = action.get("type")
        if kind == "list_files":
            listing = self.files()
            return ("\n".join(item["path"] for item in listing) or "No project files found.", None)
        if kind in {"search_files", "search_text"}:
            query = action.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"{kind} requires a non-empty query.")
            if kind == "search_files":
                matches = [item["path"] for item in self.files() if query.casefold() in item["path"].casefold()]
            else:
                matches = []
                for item in self.files():
                    if item["size"] > MAX_FILE_BYTES:
                        continue
                    try:
                        content = self.project_path(item["path"]).read_text(encoding="utf-8")
                    except (OSError, UnicodeError):
                        continue
                    matches.extend(
                        f'{item["path"]}:{line_no}: {line[:300]}'
                        for line_no, line in enumerate(content.splitlines(), 1)
                        if query.casefold() in line.casefold()
                    )
                    if len(matches) >= 100:
                        break
            return "\n".join(matches[:100]) or "No matching results.", None
        if kind == "read_file":
            value = action.get("path")
            if not isinstance(value, str):
                raise ValueError("read_file requires a string path.")
            path = self.project_path(value)
            if not path.is_file() or path.stat().st_size > 2_000_000:
                raise ValueError("The file does not exist or is too large to read.")
            content = path.read_text(encoding="utf-8")
            self.snapshots.setdefault(session_id, {})[value] = content
            return content[:30_000], None
        if kind == "write_file":
            value, content = action.get("path"), action.get("content")
            if not isinstance(value, str) or not isinstance(content, str):
                raise ValueError("write_file requires string path and content fields.")
            if len(content.encode("utf-8")) > MAX_FILE_BYTES:
                raise ValueError(f"Files must be no larger than {MAX_FILE_BYTES:,} bytes.")
            path = self.project_path(value)
            old = ""
            if path.exists():
                if not path.is_file():
                    raise ValueError("The target is not a regular file.")
                if path.stat().st_nlink > 1:
                    raise ValueError("Hard-linked files cannot be modified.")
                snapshots = self.snapshots.setdefault(session_id, {})
                if value not in snapshots:
                    raise ValueError("Read the existing file before editing it.")
                old = path.read_text(encoding="utf-8")
                if old != snapshots[value]:
                    raise ValueError("The file changed since it was read; read it again before editing.")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="")
            self.snapshots.setdefault(session_id, {})[value] = content
            diff = list(difflib.unified_diff(
                old.splitlines(), content.splitlines(),
                fromfile=f"a/{value}" if old else "/dev/null",
                tofile=f"b/{value}", lineterm="",
            ))
            change = {
                "path": value,
                "created": not bool(old),
                "added": sum(1 for line in diff if line.startswith("+") and not line.startswith("+++")),
                "removed": sum(1 for line in diff if line.startswith("-") and not line.startswith("---")),
                "diff": diff,
            }
            return f'{"Created" if not old else "Updated"} {value}.', change
        if kind == "check_python":
            value = action.get("path")
            if not isinstance(value, str) or not value.lower().endswith(".py"):
                raise ValueError("check_python requires a Python file path.")
            source = self.project_path(value).read_text(encoding="utf-8")
            ast.parse(source, filename=value)
            return f"Python syntax OK: {value}", None
        if kind == "run_tests":
            result = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-s", str(self.root),
                 "-p", "test*.py", "-v"],
                cwd=self.root, capture_output=True, text=True, timeout=120, shell=False,
            )
            output = (result.stdout + "\n" + result.stderr).strip()
            if re.search(r"\bRan 0 tests?\b", output, flags=re.IGNORECASE):
                return (
                    "Tests not verified: no tests were discovered.\n"
                    f"{output or '(no test output)'}",
                    None,
                )
            status = "passed" if result.returncode == 0 else f"failed (exit {result.returncode})"
            return f"Tests {status}.\n{output or '(no test output)'}", None
        raise ValueError(f"Unsupported action: {kind}")

    def model_catalog(self) -> dict[str, Any]:
        try:
            response = ollama.list()
            models = sorted(model.model for model in response.models)
            return {
                "connected": True,
                "models": models,
                "model": DEFAULT_MODEL,
                "model_available": DEFAULT_MODEL in models,
            }
        except Exception as exc:
            return {
                "connected": False,
                "models": [],
                "model": DEFAULT_MODEL,
                "model_available": False,
                "error": str(exc),
            }

    def respond(
        self, session_id: str, goal: str, emit: Any, model: str,
        cancel_event: threading.Event | None = None,
    ) -> None:
        deadline = time.monotonic() + self.max_task_seconds

        def stop_reason() -> str | None:
            if cancel_event is not None and cancel_event.is_set():
                return "Task cancelled."
            if time.monotonic() >= deadline:
                return "Task stopped after reaching its time limit."
            return None

        def stop(reason: str) -> None:
            emit({"type": "assistant_message", "text": reason})
            emit({"type": "complete"})

        with self.lock:
            messages = self.sessions.setdefault(session_id, [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "system",
                    "content": f"Project root: {self.root}\n"
                    "Available tools: " + ", ".join(ACTION_TYPES),
                },
            ])
        task_text = goal.casefold()
        explicitly_requested_tests = bool(re.search(
            r"\b(?:run|execute|rerun|verify)\b.{0,40}\b(?:tests?|test suite)\b",
            task_text,
        ))
        required_tools: set[str] = set()
        if explicitly_requested_tests:
            required_tools.add("run_tests")
        requires_changes = bool(re.search(
            r"\b(?:add|build|create|fix|implement|modify|refactor|replace|update|write)\b",
            task_text,
        ))
        requires_inspection = bool(re.search(
            r"\b(?:examine|inspect|review|read)\b", task_text,
        ))
        successful_edits = 0
        successful_reads = 0
        requested_python_checks = set()
        if "check_python" in task_text or "check python" in task_text:
            requested_python_checks.update(
                path.replace("\\", "/")
                for path in re.findall(r"(?<![\w.-])[\w./\\-]+\.py\b", goal, flags=re.IGNORECASE)
            )
        completed_tools: set[str] = set()
        completed_python_checks: set[str] = set()
        failed_action_counts: dict[str, int] = {}
        no_action_turns = 0
        messages.append({"role": "user", "content": goal})
        emit({"type": "user_message", "text": goal})
        emit({"type": "status", "text": f"Connecting to {model}"})
        for step in range(1, MAX_TURNS + 1):
            reason = stop_reason()
            if reason:
                stop(reason)
                return
            answer = None
            raw = ""
            for attempt in range(MAX_RESPONSE_RETRIES + 1):
                reason = stop_reason()
                if reason:
                    stop(reason)
                    return
                emit({"type": "status", "text": f"Waiting for {model} — turn {step}"})
                try:
                    stream = self.client.chat(
                        model=model, messages=messages, format=RESPONSE_FORMAT,
                        think=False, stream=True,
                    )
                    pieces = []
                    for chunk in stream:
                        reason = stop_reason()
                        if reason:
                            stop(reason)
                            return
                        if chunk.message.content:
                            pieces.append(chunk.message.content)
                except httpx.TimeoutException:
                    reason = stop_reason()
                    if reason is None:
                        raise
                    stop(reason)
                    return
                reason = stop_reason()
                if reason:
                    stop(reason)
                    return
                raw = "".join(pieces)
                try:
                    answer = json.loads(raw)
                    if not isinstance(answer, dict):
                        raise ValueError("the response must be a JSON object")
                    if not isinstance(answer.get("message"), str):
                        raise ValueError("message must be a string")
                    if not isinstance(answer.get("done"), bool):
                        raise ValueError("done must be a boolean")
                    if not isinstance(answer.get("actions"), list):
                        raise ValueError("actions must be an array")
                    if len(answer["actions"]) > MAX_ACTIONS:
                        raise ValueError(f"actions exceeds the {MAX_ACTIONS}-tool limit")
                    if any(
                        not isinstance(action, dict) or action.get("type") not in ACTION_TYPES
                        for action in answer["actions"]
                    ):
                        raise ValueError(f"each action must use a supported tool: {', '.join(ACTION_TYPES)}")
                    break
                except (json.JSONDecodeError, ValueError) as exc:
                    answer = None
                    if attempt == MAX_RESPONSE_RETRIES:
                        excerpt = raw[:300] or "[empty response]"
                        raise ValueError(
                            f"{model} returned an invalid agent response after "
                            f"{MAX_RESPONSE_RETRIES + 1} attempts ({exc}). "
                            f"Last response: {excerpt!r}"
                        ) from exc
                    emit({
                        "type": "status",
                        "text": f"Invalid response ({exc}); retrying {attempt + 1}/{MAX_RESPONSE_RETRIES}",
                    })
                    messages.extend([
                        {
                            "role": "assistant",
                            "content": raw or "[Empty response]",
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Your previous response was invalid: {exc}. Retry this turn. "
                                "Return exactly one JSON object with a string message, a boolean "
                                "done, an actions array of no more than five supported tools, and "
                                "all required tool arguments. Do not claim an action succeeded "
                                "unless its tool has actually succeeded."
                            ),
                        },
                    ])
            if answer is None:
                raise ValueError(f"{model} did not return a valid response.")
            messages.append({"role": "assistant", "content": raw})
            injected_actions = []
            if answer["done"] and "run_tests" in required_tools and "run_tests" not in completed_tools:
                if not any(action["type"] == "run_tests" for action in answer["actions"]):
                    injected_actions.append({
                        "type": "run_tests",
                        "_origin": "user_request" if explicitly_requested_tests else "verification",
                    })
            if answer["done"]:
                for path in sorted(requested_python_checks - completed_python_checks):
                    if not any(
                        action["type"] == "check_python" and action.get("path") == path
                        for action in answer["actions"]
                    ):
                        injected_actions.append({
                            "type": "check_python", "path": path, "_origin": "user_request",
                        })
            if injected_actions:
                emit({"type": "status", "text": "Executing requested verification tools"})
                answer["actions"].extend(injected_actions)
                answer["message"] = ""
                answer["done"] = False
            results = []
            failed = False
            for action in answer["actions"]:
                reason = stop_reason()
                if reason:
                    emit({"type": "assistant_message", "text": reason})
                    emit({"type": "complete"})
                    return
                kind = action["type"]
                origin = action.pop("_origin", None)
                action_key = json.dumps(action, sort_keys=True, ensure_ascii=False)
                label = action.get("path") or action.get("query") or kind.replace("_", " ")
                started = {"type": "tool_started", "tool": kind, "label": label}
                if origin:
                    started["origin"] = origin
                emit(started)
                try:
                    result, change = self.action(action, session_id)
                    success = not result.startswith((
                        "Action error:", "Tests failed (", "Tests not verified:",
                    ))
                    event = {"type": "tool_completed", "tool": kind, "label": label,
                             "result": result, "success": success}
                    failed = failed or not success
                    if success:
                        failed_action_counts.pop(action_key, None)
                        completed_tools.add(kind)
                        if kind == "read_file":
                            successful_reads += 1
                        if kind == "check_python":
                            completed_python_checks.add(action.get("path", ""))
                        if kind == "write_file":
                            successful_edits += 1
                            required_tools.add("run_tests")
                            queued_tests = any(
                                queued["type"] == "run_tests" for queued in answer["actions"]
                            )
                            if "run_tests" not in completed_tools and not queued_tests:
                                answer["actions"].append({
                                    "type": "run_tests", "_origin": "verification",
                                })
                                answer["done"] = False
                                answer["message"] = ""
                            check_path = action.get("path", "").replace("\\", "/")
                            queued_check = any(
                                queued["type"] == "check_python"
                                and queued.get("path") == check_path
                                for queued in answer["actions"]
                            )
                            if (
                                check_path in requested_python_checks
                                and check_path not in completed_python_checks
                                and not queued_check
                            ):
                                answer["actions"].append({
                                    "type": "check_python", "path": check_path,
                                    "_origin": "user_request",
                                })
                                answer["done"] = False
                                answer["message"] = ""
                    else:
                        failed_action_counts[action_key] = failed_action_counts.get(action_key, 0) + 1
                    if change:
                        event["change"] = change
                        emit({"type": "change", **change})
                except (OSError, UnicodeError, SyntaxError, ValueError,
                        subprocess.TimeoutExpired) as exc:
                    result = f"Action error: {exc}"
                    failed = True
                    failed_action_counts[action_key] = failed_action_counts.get(action_key, 0) + 1
                    event = {"type": "tool_completed", "tool": kind, "label": label,
                             "result": result, "success": False}
                emit(event)
                results.append({"action": action, "result": result})
                if failed_action_counts.get(action_key, 0) >= MAX_IDENTICAL_FAILURES:
                    message = (
                        f"Stopped after the same action failed {MAX_IDENTICAL_FAILURES} times: "
                        f"{label}. Review the error and start a new task when ready."
                    )
                    emit({"type": "assistant_message", "text": message})
                    emit({"type": "complete"})
                    return
            reason = stop_reason()
            if reason:
                stop(reason)
                return
            if failed:
                emit({"type": "status", "text": "A tool failed; asking the model to recover"})
                messages.append({
                    "role": "user",
                    "content": "Tool results:\n" + json.dumps(results, ensure_ascii=False),
                })
                messages.append({
                    "role": "user",
                    "content": "At least one tool action failed. The exact results are above. "
                    "The task is NOT complete. Correct the invalid action arguments and retry. "
                    "Do not claim success unless the tool succeeds.",
                })
                continue
            if answer["done"]:
                missing_requirements = []
                if requires_inspection and successful_reads == 0:
                    missing_requirements.append("read at least one relevant project file")
                if requires_changes and successful_edits == 0:
                    missing_requirements.append("make and verify the requested project changes")
                if missing_requirements:
                    reason = "The task is not complete: " + " and ".join(missing_requirements) + "."
                    emit({"type": "status", "text": reason + " Asking the model to continue"})
                    messages.extend([
                        {
                            "role": "assistant",
                            "content": json.dumps(answer, ensure_ascii=False),
                        },
                        {
                            "role": "user",
                            "content": "Tool results:\n" + json.dumps(
                                results, ensure_ascii=False,
                            ),
                        },
                        {
                            "role": "user",
                            "content": reason + " Continue using the project tools. Do not "
                            "claim completion until these requirements have actually succeeded.",
                        },
                    ])
                    continue
            if answer["message"]:
                emit({"type": "assistant_message", "text": answer["message"]})
            if answer["done"]:
                with self.lock:
                    self.sessions[session_id] = messages[-32:]
                emit({"type": "complete"})
                return
            if not results:
                no_action_turns += 1
                if no_action_turns >= MAX_NO_ACTION_TURNS:
                    emit({
                        "type": "assistant_message",
                        "text": (
                            f"Stopped after {MAX_NO_ACTION_TURNS} consecutive turns without "
                            "taking an action. Start a new task with a more specific request."
                        ),
                    })
                    emit({"type": "complete"})
                    return
                results = [{"result": "No action taken. Choose a useful tool or finish."}]
            else:
                no_action_turns = 0
            messages.append({"role": "user", "content": "Tool results:\n" + json.dumps(results)})
        emit({"type": "assistant_message", "text": f"Stopped after {MAX_TURNS} model turns; start a new task to continue."})
        emit({"type": "complete"})


class Handler(BaseHTTPRequestHandler):
    server_version = "VSCodeAIHarness/1.0"

    @property
    def harness(self) -> Harness:
        return self.server.harness  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.address_string()}] {format % args}")

    def _json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            self._json({**self.harness.model_catalog(), "root": str(self.harness.root),
                        "files": self.harness.files()})
            return
        if path == "/" or path in {"/index.html", "/styles.css", "/app.js"}:
            name = "index.html" if path == "/" else path.lstrip("/")
            file = Path(__file__).parent / "static" / name
            if not file.is_file():
                self._json({"error": "Application asset not found."}, 404)
                return
            content_type = "text/html; charset=utf-8" if name.endswith(".html") else (
                "text/css; charset=utf-8" if name.endswith(".css") else "text/javascript; charset=utf-8"
            )
            body = file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/file":
            querystring = urlparse(self.path).query
            params = dict(item.split("=", 1) for item in querystring.split("&") if "=" in item)
            file_path = params.get("path")
            if not file_path or not file_path.startswith("/"):
                self._json({"error": "A non-empty path parameter is required."}, 400)
                return
            try:
                project_rel_path = self.harness.project_path(file_path.removeprefix("/"))
            except ValueError as exc:
                error_msg = str(exc).removeprefix("Action error: ").strip()
                self._json({"error": error_msg}, 403)
                return
            if not project_rel_path.is_file():
                self._json({"error": f"The file {file_path} does not exist."}, 404)
                return
            try:
                size = project_rel_path.stat().st_size
            except OSError:
                self._json({"error": f"Could not stat {file_path}."}, 500)
                return
            if size > 1_000_000:
                self._json({"error": f"The file {file_path} exceeds 1 MB."}, 413)
                return
            content = project_rel_path.read_text(encoding="utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))
            return
        self._json({"error": "Not found."}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/api/chat", "/api/cancel"}:
            self._json({"error": "Not found."}, 404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_RESPONSE_BYTES:
                raise ValueError("Request body must be between 1 byte and 2 MB.")
            data = json.loads(self.rfile.read(size))
            goal = data.get("message")
            session_id = data.get("session_id")
            model = data.get("model", DEFAULT_MODEL)
            if not isinstance(session_id, str) or len(session_id) > 80:
                raise ValueError("Invalid chat session.")
            if path == "/api/cancel":
                self._json({"cancelled": self.harness.cancel_task(session_id)})
                return
            if not isinstance(goal, str) or not goal.strip():
                raise ValueError("Enter a message before sending.")
            if not isinstance(model, str) or model not in self.harness.model_catalog()["models"]:
                raise ValueError(f"{model or 'Selected model'} is not installed in Ollama.")
            cancel_event = self.harness.register_task(session_id)
        except (ValueError, json.JSONDecodeError, AttributeError) as exc:
            self._json({"error": str(exc)}, 400)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def emit(event: dict[str, Any]) -> None:
            self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()

        try:
            self.harness.respond(session_id, goal.strip(), emit, model, cancel_event)
        except (ollama.RequestError, ollama.ResponseError, ConnectionError, OSError,
                ValueError, json.JSONDecodeError, httpx.TimeoutException) as exc:
            try:
                emit({"type": "error", "message": f"Ollama or agent error: {exc}"})
                emit({"type": "complete"})
            except (BrokenPipeError, ConnectionResetError):
                pass
        finally:
            self.harness.finish_task(session_id, cancel_event)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local VS Code-style Ollama AI harness.")
    parser.add_argument("--root", default=".", help="Project folder to give the agent access to")
    parser.add_argument("--host", default="127.0.0.1", help="Web app bind address")
    parser.add_argument("--port", type=int, default=8765, help="Web app port")
    parser.add_argument(
        "--max-task-hours", type=float, default=MAX_TASK_HOURS,
        help=f"Maximum runtime for each agent task (default: {MAX_TASK_HOURS} hours)",
    )
    args = parser.parse_args()
    if not math.isfinite(args.max_task_hours) or args.max_task_hours <= 0:
        parser.error("--max-task-hours must be a finite number greater than zero.")
    root = Path(args.root).resolve()
    try:
        workspace = ensure_workspace(root)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.harness = Harness(workspace, max_task_seconds=args.max_task_hours * 3600)  # type: ignore[attr-defined]
    print(f"VS Code AI Harness: http://{args.host}:{args.port}/")
    print(f"Agent workspace: {workspace}")
    print(f"Default model: {DEFAULT_MODEL}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping application.")
    finally:
        server.server_close()
        server.harness.client.close()  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
