# Architecture and Mermaid flowcharts

Render the following blocks in any Mermaid-compatible Markdown preview.

## 1. Student placement prediction

```mermaid
flowchart TD
    A[CSV selected in browser] --> B[Local web server validates request size and JSON]
    B --> C[Write CSV to isolated temporary directory]
    C --> D[predict.py validates required columns, IDs, and target absence]
    D --> E[Load placement_model.joblib and model_config.json]
    E --> F[Pipeline imputes, scales, and encodes features]
    F --> G[Logistic regression probabilities]
    G --> H[Apply saved threshold: 0.59]
    H --> I[Write student_id, placed CSV]
    I --> J[Browser shows table and counts]
    J --> K[Download predictions CSV]
    D -. invalid input .-> L[Return useful validation error]
```

The fitted preprocessing and classifier travel together in the saved pipeline. `student_id` is carried to output; `status_code` is validated as an input column but omitted from model features because its timing and meaning are undocumented.

## 2. Class message task extraction

```mermaid
flowchart TD
    A[Select TXT files in browser] --> B[Local web server checks count and byte limits]
    B --> C[Ignore client filenames; assign safe temporary filenames]
    C --> D[extract.py parses headers and message body]
    D --> E[Trusted system prompt plus JSON-quoted untrusted body]
    E --> F[Ollama at 127.0.0.1:11434]
    F --> G[Local Qwen 2.5 3B model]
    G --> H[Parse JSON and enforce exact intermediate schema]
    H --> I[Ground owner and deadline in source text]
    I --> J[Override obvious request drift; exclude clear non-actions]
    J --> K[Write exact tasks array JSON]
    K --> L[Browser renders tasks and diagnostics]
    L --> M[Download tasks.json]
    H -. malformed, timeout, or schema error .-> N[Record per-message failure and continue]
    N --> K
```

Message contents are sent only to the loopback Ollama API. The prompt and validation controls reduce prompt-injection and unsupported-claim risks; they do not guarantee semantic correctness.

## 3. Local web application and trust boundaries

```mermaid
flowchart LR
    subgraph Browser[Browser on this computer]
        UI[Separate Overview, Placement, Messages, Architecture pages]
        CSV[Student CSV file]
        TXT[Message TXT files]
    end
    subgraph LocalHost[127.0.0.1 only]
        HTTP[Python ThreadingHTTPServer]
        PRE[Placement predictor and saved artifacts]
        EXT[Message extractor subprocess]
        OLLAMA[Ollama local API]
        MODEL[Local Qwen model weights]
    end
    CSV --> UI
    TXT --> UI
    UI -->|JSON request| HTTP
    HTTP -->|temporary CSV| PRE
    PRE -->|prediction records| HTTP
    HTTP -->|temporary message files| EXT
    EXT -->|loopback JSON chat| OLLAMA
    OLLAMA --> MODEL
    MODEL --> OLLAMA
    OLLAMA --> EXT
    EXT -->|validated tasks and diagnostics| HTTP
    HTTP -->|JSON response| UI
    UI -->|download generated file| Browser
    X[Hosted inference services] -. no connection .- UI
```

The development server binds only to loopback addresses. It does not accept filesystem paths or model names from the browser, ignores uploaded filenames for disk placement, imposes upload limits, and serves a restrictive Content Security Policy.
