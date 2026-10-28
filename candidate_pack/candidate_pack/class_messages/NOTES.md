# Class message task extraction

## Model and local setup

The client defaults to `qwen2.5:3b-instruct-q4_K_M` through Ollama's local HTTP interface at `127.0.0.1:11434`. The [Ollama model library](https://ollama.com/library/qwen2.5/tags) lists this quantization at about 1.9 GB with a 32K context; this implementation caps context at 2,048 tokens and generation at 256 tokens. It is a compact choice for a 16 GB Windows machine. The [official Windows setup](https://ollama.com/download/windows) supports local Ollama installation. No message text is sent to a hosted inference service.

The prior work record says Ollama and the Qwen weights were not installed in its preparation environment, and external network access was unavailable. A cached `google/flan-t5-base` (about 990 MB) was also evaluated locally, but it produced malformed JSON for the solved example and labeled a non-task tea invitation actionable, so it was rejected. Ollama 0.40.12 and the selected Qwen model were subsequently installed and verified in the model-backed runs below.

Install Ollama from its official Windows installer (the verified machine run used Ollama 0.40.12). In PowerShell, `Get-Location` shows the current folder, and `Get-Command ollama` shows the CLI path if it is on `PATH`. The verified installation's CLI is at `$env:LOCALAPPDATA\Programs\Ollama\ollama.exe`; if it is not on `PATH`, invoke it by full path. Then, from this `class_messages` directory:

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull qwen2.5:3b-instruct-q4_K_M
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" list
python extract.py --inbox messages --output tasks.json
python extract.py --inbox example --output example_tasks.json
```

Ollama should be running locally before the commands. Python dependencies are standard-library only; `requirements.txt` is intentionally empty of third-party runtime packages.

## Prompt and output

Each file's headers are parsed locally; only its body is sent to the model as a JSON-quoted untrusted string. A trusted system message defines actionable requests, owner/deadline rules, and the exact intermediate JSON schema. It explicitly treats obligation wording such as “Arjun needs to…” as actionable. For clear opening forms (“Name, please…”, “Name needs to…”, “Please…”, “Could someone…”), a conservative source-based fallback corrects model drift or an injection-induced result. It derives common date/time phrases and short “by/before” deadlines directly from the original text. Clear no-action notices and open-ended “anyone up for” social invitations are excluded even if the model invents a task or owner. Temperature is 0 and seed is 42. The final file has exactly the required top-level `tasks` array; actionable records contain `message_id`, `task`, `owner`, and `due`. Non-actionable messages are omitted. Deadline wording is preserved from the exact source substring, even when the model normalizes spacing or punctuation; owner and deadline text are removed from the task line.

The client validates JSON, exact keys, value types, actionable/non-actionable consistency, owner presence, and deadline presence before adding a task. It never executes model text. The API URL is fixed to loopback rather than accepting a model-supplied destination. Invalid or failed messages are reported to stderr, excluded, and do not stop processing other files. A partial run still writes valid JSON and exits with status 2 so it cannot be mistaken for a fully successful run.

These controls reduce injection and fabrication risk; they do not guarantee semantic correctness. The semantic quality of tasks still depends on the local model. The supplied adversarial text is untrusted and must not change the extraction rules.

## Tokens, performance, and determinism

On 28 Oct 2026, two consecutive runs using Ollama 0.40.12 and `qwen2.5:3b-instruct-q4_K_M` each processed all six practice messages with four tasks (`m01`, `m03`, `m04`, `m05`) and zero failures. Both reported 2,326 prompt tokens; output counts were 182 and 176 tokens, and elapsed times were 24.426 and 20.297 seconds. The output files had identical SHA-256 hashes (`DD9CB9774775BF036074418904AE43B0564FA947BED3E0B4212D04E90D377BA0`) and were byte-for-byte identical. A third run, initiated from PowerShell, also processed all six messages with four tasks and zero failures (2,326 prompt tokens, 204 output tokens, 27.645 seconds). The latest instrumented run recorded actual usage per input message:

| Message ID | Prompt tokens | Output tokens | Elapsed seconds | Outcome |
|---|---:|---:|---:|---|
| m01 | 382 | 34 | 13.782 | task |
| m02 | 383 | 34 | 3.336 | no task |
| m03 | 388 | 39 | 3.624 | task |
| m04 | 411 | 40 | 4.190 | task |
| m05 | 385 | 36 | 3.357 | task |
| m06 | 377 | 21 | 2.000 | no task |
| **Total** | **2,326** | **204** | **30.294** | **4 tasks, 0 failures** |

The prompt count is Ollama's `prompt_eval_count` for the complete per-message prompt (system instructions plus the message payload); output count is `eval_count` for the generated response. The command now prints a JSON usage record per input message and a batch total to stderr. The frontend shows the same usage beside each extracted task and in a complete per-message table, including messages with no task. Counts are not inserted into `tasks.json`, preserving its exact required schema. A run on the solved example processed one message with one task and zero failures (381 prompt tokens, 33 output tokens, 2.932 seconds) and matched `example/expected.json`. Temperature zero and a fixed seed support repeatability but do not guarantee identical behavior across runtime/model revisions.

An end-to-end POST through the running local frontend also processed the same six messages with four tasks and no partial result. It received the same per-message token counts (2,326 prompt and 204 output tokens total) and took 18.005 seconds in that run. Per-message elapsed time varies with model warm-up and system load.

## Tested behavior and limitations

Run the offline contract suite with `python -m unittest discover -s tests -v` from this folder. It covers the solved example schema with mocked inference, prompt structure and deterministic settings, malformed JSON, extra keys, invalid types, unsupported owners, due normalization, fallback for `m03` and `m04`, exclusion of m02, injection override, continuing after a per-message failure, Windows CRLF separators, and invalid UTF-8 isolation. A fixed response stub also produced expected task IDs `m01`, `m03`, `m04`, and `m05` from the six practice messages; the injected answer-key task was absent. These are code/schema tests, not semantic results from Qwen.

The model-backed practice folder was run multiple times and the solved example once as recorded above. The offline contract suite passed all 16 tests, including Windows line-ending, invalid-encoding, per-message token accounting, and output-schema cases. The script has no hosted fallback. Inference is local to the user's Ollama service; safeguards reduce prompt-injection and fabrication risk but cannot guarantee semantic correctness.
