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

From this directory, launch the app with your selected project as its workspace:

```powershell
python app.py --root "C:\path\to\your\project"
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). To choose another port,
pass `--port 9000`. Only bind to the local machine: the application gives the
connected model the ability to read and write files in the selected workspace.

To open the application folder as its own workspace:

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
Sensitive files and paths outside the selected workspace are rejected. The
assistant does not run arbitrary shell commands.

## Test

```powershell
python -m unittest discover -s . -p "test*.py" -v
```
