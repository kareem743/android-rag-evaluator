# RAG Evaluator

Evaluate an Android app on a phone or emulator, or another HTTP RAG pipeline.
The target application performs retrieval and generation. Python sends test
questions to its API, optionally scores its answers and retrieved evidence using
a model API, and saves JSON/JSONL reports.

The judge accepts an API key, model ID, and configurable base URL using the
OpenAI-compatible Chat Completions protocol. No local judge installation or
model download is required.

## Install

```powershell
git clone https://github.com/kareem743/android-rag-evaluator.git
cd android-rag-evaluator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r tools/eval/requirements.txt
```

Use Python 3.10 or newer. If PowerShell blocks activation, replace `python` in
subsequent commands with `.\.venv\Scripts\python.exe`. The standalone repository
contains only the evaluator: supply your own application, corpus, and test dataset.

## Connect your application

For an Android phone, enable USB debugging and launch the app's debug build.
For an emulator, start a virtual device in Android Studio and launch the same
app. Its debug API must already be implemented by the app.

```text
adb devices
adb -s DEVICE_SERIAL forward tcp:9000 tcp:9000
```

Replace `DEVICE_SERIAL` with the serial shown by `adb devices` (for example,
`emulator-5554`). Use the application's actual port on the right. An HTTP RAG
server needs no ADB; pass its URL with `--api-base`.

### Common API format

Select `--adapter generic`. The app must accept POST `/evaluate`:

```json
{"case_id": "case-1", "question": "What does the guide say?", "top_k": 3,
 "metadata": {"category": "", "notes": ""}}
```

Return the answer and the retrieved chunks actually used, in retrieval order:

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

Answer and chunks are required for answer evaluation; timing and runtime
metrics are optional. Empty chunk lists are allowed. The generic adapter needs
no health endpoint or BRITE-specific readiness fields.

### Map another API

Copy [profiles/other_android.example.json](profiles/other_android.example.json)
or download a profile from the website's **App profiles** tab. Set the endpoint,
request fields, response fields, and chunk fields to match your app, then use:

```text
python tools/eval/client.py --adapter-profile my_app.json --api-base http://127.0.0.1:9000 --dataset my_cases.json --no-judge
```

The example profile uses `/ask`, `query`, `options.limit`, `/status`, and nested
response fields such as `result.answer`. Dotted paths support nested objects;
array indexing is not supported. Optional fields can be set to `null`.
Unsupported top-k, model switching, and retrieval-only overrides fail clearly.

`judge_domain` controls the judging prompt (for example `general`, `software
documentation`, or `medical first-aid`); `--judge-domain` overrides it.
`headers_env` maps **application** header names to environment variable names.
For example `{"Authorization": "RAG_AUTH"}` reads a full header value from
`RAG_AUTH`. This is separate from the judge API key. Do not put keys in profiles.

The BRITE adapter is still available with `--adapter brite` (also the default
for existing commands). It retains BRITE's health checks and corpus upload
support. For another app, index its corpus through that application's own tools.

## Supply a test dataset

Save a JSON array as `my_cases.json`, or use one JSON object per line in JSONL:

```json
[
  {
    "case_id": "case-1",
    "question": "Your test question",
    "ground_truth_answer": "The expected answer",
    "category": "optional"
  }
]
```

The website can download this template. The CLI's legacy BRITE dataset default
is not included in this standalone repository, so always supply `--dataset`.

## Run without scoring first

```text
python tools/eval/client.py --adapter generic --api-base http://127.0.0.1:9000 --dataset my_cases.json --limit 3 --no-judge --output tools/eval/output/my_run.jsonl
```

This checks the app connection and saves its answers and evidence. Scores remain
unavailable. `--retrieval-only` also disables judging and requires support from
the target app.

## Configure API judging

Set your key for the current terminal session. This PowerShell example masks
input and keeps the key out of command history:

```powershell
$judgeSecret = Read-Host "Judge API key" -AsSecureString
$env:RAG_JUDGE_API_KEY = [System.Net.NetworkCredential]::new("", $judgeSecret).Password
```

Choose the provider's base URL and a Chat Completions-compatible model ID from
your account. The evaluator appends `/chat/completions` to the base URL.

| API | Judge base URL | Reference |
| --- | --- | --- |
| OpenAI | `https://api.openai.com/v1` | [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) |
| Gemini compatibility API | `https://generativelanguage.googleapis.com/v1beta/openai` | [Compatibility guide](https://ai.google.dev/gemini-api/docs/openai) |
| Claude compatibility API | `https://api.anthropic.com/v1` | [Compatibility guide](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk) |
| Another compatible API | Its documented API base URL | Requires Bearer authentication and Chat Completions message responses |

Native APIs with a different protocol need a compatible gateway; this transport
does not translate every vendor's native API. Claude's compatibility layer also
requires a key scoped to one workspace for this simple configuration; keys that
require an extra workspace header need a compatible gateway.

```text
python tools/eval/client.py --adapter generic --api-base http://127.0.0.1:9000 --dataset my_cases.json --judge-api-base https://api.openai.com/v1 --judge-model MODEL_ID --output tools/eval/output/scored_run.jsonl
```

Replace `MODEL_ID` with your actual model ID. For a custom app, replace
`--adapter generic` with `--adapter-profile my_app.json`.

Settings:

- `--judge-api-base` / `--judge-base-url`: API base URL. Also reads `RAG_JUDGE_BASE_URL`; defaults to the OpenAI API URL.
- `--judge-model`: model ID. Also reads `RAG_JUDGE_MODEL`; no model is hardcoded.
- `--judge-api-key-env`: key environment variable name; defaults to `RAG_JUDGE_API_KEY`.
- `--judge-timeout`: request timeout in seconds; defaults to 180.
- `--no-judge`: disable scoring; no judge key or model is needed.

The key is sent only in the judge request's Authorization header, and excluded
from configuration previews, reports, and logs. Judge requests use HTTPS;
loopback HTTP is supported for local compatible servers and testing. Automatic
redirects are disabled. Questions, reference answers, generated answers, and
retrieved evidence are sent to the selected judge provider. Hosted judging needs
an internet connection and uses that provider's API billing.

API failures, timeouts, and malformed judgments produce unscored error rows and
allow other cases to finish. Missing judge settings fail before the run starts.

## Local Python dashboard

```text
python tools/eval/dashboard.py
```

Open `http://127.0.0.1:8050`. Choose **Generic Android / HTTP API** or upload an
application profile; upload your dataset and set the application's API URL.
For another app, use **Model ID Override** instead of the BRITE model dropdown.
Leave unsupported controls empty.

Enable **Use API Judge**, enter **Judge API Base URL**, **Judge Model**, and the
masked **Judge API Key**. Alternatively set `RAG_JUDGE_API_KEY` before starting
Python and leave the key field empty. The environment key is never embedded in
the page layout, and the input is not persisted to browser storage. Disable API
judging to collect outputs without a key.

## Reports and interpretation

Both CLI and dashboard write JSON and JSONL. Open either file in the hosted
website to inspect answers, evidence, timings, and scores. The website remains
a report viewer and profile builder; it does not run Python or collect API keys.
Reports are processed in browser memory and cleared on refresh.

The judge assigns chunk relevance labels and answer scores. These are model
judgments, not human ground truth. Metrics evaluate the top five chunks. NDCG
uses the ideal ordering of those returned chunks, rather than corpus-wide recall.
This API transport change preserves the scoring rules and report row format.
There is no automatic parameter optimization or production-query monitoring.

## Tests

```text
python -m pip install -r tools/eval/requirements-dev.txt
python -m pytest tools/eval -q
```

Tests cover application adapters, a simulated target app and judge over real
HTTP, authentication, configuration, timeouts, unavailable scores, key redaction,
CLI reports, dashboard controls, and continued execution after failed cases.
