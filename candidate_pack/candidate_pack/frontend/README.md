# GoML Engineer local frontend

A minimal, local-only multi-page web interface for the two candidate-pack workflows. Overview, Placement, Messages, and Architecture are separate pages with normal browser navigation. It uses the fitted placement model and extraction script; it does not introduce another model or hosted inference dependency.

## Start on Windows

Open PowerShell and run:

```powershell
Set-Location "F:\GoML Engineer\candidate_pack\candidate_pack\placement"
python -m pip install -r requirements.txt
Set-Location "F:\GoML Engineer\candidate_pack\candidate_pack\frontend"
python server.py
```

Open `http://127.0.0.1:8000` in a browser. Stop the server with **Ctrl+C** in PowerShell. The terminal shows its current directory in the prompt; use `Get-Location` to print it.

Ollama must be running locally and the configured model must be present. Verify in PowerShell with:

```powershell
ollama --version
ollama list
```

The frontend checks the Ollama API and disables message extraction until `qwen2.5:3b-instruct-q4_K_M` is available. Placement scoring can run independently.

## Features

- Placement: upload a CSV, validate and score it with `placement/predict.py`, inspect counts and rows, then download `placement_predictions.csv`. The currently fitted logistic-regression model uses C=0.3 and threshold 0.59.
- Messages: upload up to 50 `.txt` files, process them through the local model, review tasks, per-message prompt/output token counts and elapsed time (including non-actionable messages), review partial-failure diagnostics, then download `tasks.json`.
- Files are held in temporary directories while the request runs. The server binds only to loopback, accepts no client-supplied disk paths, and uses bounded upload sizes.
- No analytics, remote scripts, or hosted model calls are used.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for Mermaid flowcharts for placement prediction, task extraction, and the local app trust boundaries.

## Limits

This is a single-user local utility, not an authenticated multi-user service. It has no persistent database or job queue. Large message batches can take several minutes on CPU. A partial extraction result is labeled and carries diagnostics; semantic extraction still depends on local model behavior.
