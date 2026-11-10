# VS Code AI Harness

A standalone, locally served VS Code-inspired coding assistant. It connects to
Ollama, defaults to `qwen3.5:9b`, shows actual model tool calls and workspace
changes, and needs no JavaScript build or frontend package installation.

## Requirements

- Python 3.10 or later
- Ollama running locally
- The Ollama Python package (`pip install ollama`)
- The default model (`ollama pull qwen3.5:9b`)

## Run

Launch the app with the project folder that should contain the agent's workspace:

```powershell
python app.py --root "C:\path\to\your\project"
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). To choose another port,
pass `--port 9000`. The app creates `workspace` inside the selected project
folder if it does not exist. The agent can only list, read, and modify files in
that subfolder; files elsewhere in the selected project are not exposed through
its file tools or file viewer. Tests are also discovered and run from this
workspace. Only bind to the local machine.
Each task stops after eight hours by default; adjust the limit with
`--max-task-hours 6`. A model response that makes no progress for 60 seconds
times out, identical failed actions stop after two attempts, and three
consecutive turns without an action stop the task. Each request is also capped
at 20 model turns. Use **Stop task** in the toolbar to cancel a running request.

Before leaving a task unattended, use a dedicated Git branch and commit or
stash your current work. The assistant can edit files directly in the selected
workspace; these limits prevent runaway loops but are not a substitute for
reviewing changes.

To use a `workspace` subfolder inside the application folder:

```powershell
python app.py --root .
```

The model selector lists installed Ollama models. If Ollama is stopped or
`qwen3.5:9b` is missing, the interface reports the connection state; start
Ollama and/or run `ollama pull qwen3.5:9b` before submitting work.

## What it can do

- Explore and search the selected project
- Read files and create/update files in the selected workspace
- Check Python syntax and run Python `unittest` discovery
- Display actual successful tool calls and unified diffs for model-created edits
- Continue a conversation in the current browser tab

Existing files must be read first and are checked for changes before overwrite.
Sensitive files, symbolic links, junctions, hard-linked write targets, and paths
outside `workspace` are rejected. The assistant does not provide an arbitrary
shell command tool. Python tests in the workspace still run with the same OS
permissions as the app, so this is an application-level boundary, not an
operating-system sandbox.

## Test

```powershell
python -m unittest discover -s . -p "test*.py" -v
```
