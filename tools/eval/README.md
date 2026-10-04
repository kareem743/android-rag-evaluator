# Python Evaluation Workflow (Debug)

This folder evaluates Android RAG applications through their HTTP debug APIs.
The existing BRITE app is the default; other apps can use a configurable adapter.

- `client.py`: CLI batch evaluator
- `dashboard.py`: Dash UI for running and inspecting evals
- `judge.py`: Python-side judging logic
- `common.py`: dataset/corpus parsing + Android API client helpers
- `adapters.py`: application endpoints, request/response mapping, readiness checks
- `runner.py`: shared batch execution and per-case error handling

Android remains the real RAG runtime. Python calls Android's debug API and evaluates returned outputs.

The Python evaluator does not require the target app to use BRITE's retrieval
implementation, vector store, model engine, or medical corpus. It needs the app
to expose answers and the retrieved evidence used for those answers.

## 1) Start Android App + Port Forward (PowerShell, exact)

```powershell
.\gradlew.bat installDebug

$adb = 'C:\Users\Mamoun\AppData\Local\Android\Sdk\platform-tools\adb.exe'
& $adb devices
& $adb shell am start -n io.brite.medrag/io.brite.medrag.activity.MainActivity
& $adb forward tcp:9000 tcp:9000

Invoke-RestMethod http://127.0.0.1:9000/health
```

## 2) Install Python Dependencies

```powershell
python -m pip install -r tools/eval/requirements.txt
```

## 3) Run CLI Evaluator

```powershell
python tools/eval/client.py --dataset app/src/debug/assets/eval_cases.json --api-base http://127.0.0.1:9000
```

Retrieval-only mode keeps Android on the real retrieval path and skips answer generation:

```powershell
python tools/eval/client.py --dataset app/src/debug/assets/eval_cases.json --api-base http://127.0.0.1:9000 --retrieval-only --no-ollama-judge
```

Optional corpus override. The file is streamed to Android's debug API, then Android imports it
through the same `RagRepository.createRagFromFile` path used by manual in-app RAG creation:

```powershell
python tools/eval/client.py --dataset app/src/debug/assets/eval_cases.json --api-base http://127.0.0.1:9000 --corpus C:\path\to\corpus.txt
```

## 4) Run Dashboard

```powershell
python tools/eval/dashboard.py
```

Open:

```text
http://127.0.0.1:8050
```

## 5) Connect another Android application

The other app must provide a JSON HTTP endpoint in its debug build. This tool
does not inspect an arbitrary APK's internal retrieval or modify an app for you.
Implement the endpoint alongside that app's own RAG pipeline, and return the
chunks actually used to produce the answer in retrieval order.

Bind the debug server to device loopback and forward its port through ADB:

```text
adb forward tcp:9000 tcp:9000
```

Use the app's actual server port on the right if it differs. Keep this debugging
interface disabled in release builds. No internet connection is required when
both the evaluator and the Ollama judge run locally.

### Option A: use the common response format

Accept a POST to `/evaluate`, for example:

```json
{"case_id": "case-1", "question": "What does the guide say?", "top_k": 3,
 "metadata": {"category": "", "notes": ""}}
```

Return:

```json
{
  "generated_answer": "The guide recommends ...",
  "retrieved_chunks": [
    {"text": "Evidence from the guide ...", "source": "Guide", "score": 0.9}
  ],
  "retrieval_timing_ms": 20,
  "generation_timing_ms": 100,
  "total_latency_ms": 120
}
```

Answer and retrieved chunks are required for answer evaluation. Timings and
runtime diagnostics are optional. An empty chunk list is allowed; a missing
chunk field is treated as an integration error. The generic adapter does not
require a health endpoint or BRITE's `runtime_ready` / LiteRT-LM fields.

```text
python tools/eval/client.py --adapter generic --api-base http://127.0.0.1:9000 --dataset path/to/cases.json --no-ollama-judge
```

Remove `--no-ollama-judge` to enable semantic scoring. Start Ollama separately
and ensure the selected judge model is installed; override with `--ollama-model`.

### Option B: map the other app's existing API

Copy [profiles/other_android.example.json](profiles/other_android.example.json)
and edit it to match the app. That example accepts `/ask` requests containing
`query` and `options.limit`, reads `/status` with `{"ready": true}`, and maps:

| Evaluator field | Example app field |
| --- | --- |
| Generated answer | `result.answer` |
| Retrieved chunks | `result.documents` |
| Chunk text | `body` |
| Chunk source | `metadata.title` |
| Chunk score | `similarity` |
| Total latency | `timings.total_ms` |

```text
python tools/eval/client.py --adapter-profile path/to/my_app.json --api-base http://127.0.0.1:9000 --dataset path/to/cases.json --top-k 3 --no-ollama-judge
```

Profile fields:

- `evaluate_path`: POST endpoint, relative to `--api-base`.
- `health_path`: optional GET endpoint; use `null` to skip it.
- `health_expect`: required health field values, using dotted object paths.
- `request_fields`: maps canonical request names to the app's field paths.
- `response_fields`: maps evaluator fields to response paths.
- `chunk_fields`: maps `text`, `source`, and `score` within each chunk.
- `extra_payload`: constant JSON values to include in each request.
- `judge_domain`: domain used in the judging prompt, such as `general`,
  `software documentation`, or `medical first-aid`.
- `headers_env`: optional map of HTTP header names to environment variable
  names. For example, `{"Authorization": "RAG_AUTH"}` reads the full header
  value from `RAG_AUTH`. Do not put credentials in profile files.

Dotted paths address nested JSON objects; array indexing is not supported.
Unspecified request and response mappings retain their common-format defaults.
Set a field to `null` to disable it. `question` and `retrieved_chunks` cannot be
disabled. Unsupported top-k, model, or retrieval-only overrides fail clearly
instead of silently running a different experiment.

The example disables model switching and retrieval-only mode. Implement these
features in the other app and map `model_id` / `retrieval_only` to enable them.
Use `--model-id` for an app-specific model identifier. Corpus uploading remains
BRITE-specific; index the other app's corpus inside that app before evaluation.

### Dashboard

Choose **Generic Android / HTTP API** for the common format, or upload an API
profile; a successful upload automatically selects **Uploaded API profile**.
Set the application's API URL, upload the dataset, and run the evaluation.
For other apps, use **Model ID Override** rather than the BRITE model dropdown.
Leave unsupported controls empty (including Top K Override when disabled).
An invalid profile upload blocks custom-adapter runs until a valid profile is
uploaded. Select the BRITE adapter to return to the original app.

The dashboard and CLI use the same runner. API errors, malformed responses,
timeouts, and judge failures are recorded per case so other cases can finish.
JSON/JSONL reports retain the adapter name, judging domain, requests, normalized
responses, and—for custom adapters—the original API responses. Missing runtime
diagnostics remain unavailable rather than being fabricated.

## 6) Scope and metric interpretation

This is a batch evaluator with manual top-k/model overrides. It does not add
automatic parameter optimization or monitoring of production user queries.
Retrieval-only mode records evidence and timings but currently disables semantic
judging, so it does not produce judged retrieval scores.

The judge assigns chunk relevance labels and answer scores; these are model
judgments, not human ground truth. The existing metrics evaluate the top five
chunks. NDCG uses an ideal ordering of the returned top five chunks, so it
measures ranking within that set and does not measure corpus-wide recall. These
metric semantics are preserved by this adapter refactor.

The Dash dashboard remains a local Python server. This change does not deploy
it to Cloudflare or publish the private repository as open source.

## 7) Tests

```text
python -m pip install -r tools/eval/requirements-dev.txt
python -m pytest tools/eval -q
```

Tests cover BRITE compatibility, a simulated second Android app's different
API contract over real HTTP, CLI reports, profile validation, dashboard selection,
unsupported controls, timeouts, and continued execution after failed cases.
