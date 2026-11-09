---
name: timeline-extraction-cli
description: Use when preparing MATRES/TimeML corpora, running timeline-extraction classification or document-level graph experiments, selecting account-backed Antigravity/Codex/Claude or API providers, enabling LangSmith tracing, or evaluating cached responses.
---

# Timeline Extraction CLI

Operate this repository's CLI with `uv`. Run commands from the repository root. Read [README](../../../README.md) and [CLI usage](../../../examples/cli_usage_examples.md) for configuration and artifact details; inspect subcommand `--help` before using unfamiliar flags.

## Choose the experiment

- `classification`: pairwise temporal-relation prediction.
- `consistency`: document-level temporal graph extraction and evaluation (`multi` mode).
- `comparison`: both modes with the same corpus, model, and prompting controls.
- These evaluate LLM predictions. They are not a LangGraph application or the paper's complete encoder-assisted cycle-resolution workflow.

## Choose authentication before a provider

Use the requested account or API route; never silently substitute API billing for subscription access.

| Route | Provider | Authentication |
| --- | --- | --- |
| Antigravity account | `antigravity` | Install `agy`; run `rtk agy`, sign in, then exit. List model slugs with `rtk agy models`. Default: `gemini-4-argon-high`. |
| ChatGPT account | `codex` | Install Codex; run `rtk codex login` and select ChatGPT. Default: `gpt-5.6-sol`. |
| Claude subscription | `claude` | Install Claude Code; run `rtk claude auth login` without `--console`. Default: `sonnet`. |
| OpenAI API | `openai` | `OPENAI_API_KEY` |
| Gemini API | `google` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| Together API | `together` | `TOGETHER_API_KEY` |
| Groq API | `groq` | `GROQ_API_KEY` |
| Local Transformers | `huggingface` | Optional `HF_TOKEN` for gated models; CPU by default. |

- Keep credentials in the environment or native CLI credential store. Never request keys in chat, print them, put them in command arguments, or commit them.
- Account providers reject `api_key`, `base_url`, and nonzero `temperature`. They use native login, not extracted tokens. Missing CLI/login is a prerequisite failure, not permission to substitute another backend.
- Prefer supported native LangChain integrations when they support the requested authentication. Standard API-key integrations do not reuse subscription login; the private, experimental Codex OAuth wrapper is not a replacement for official CLI access.

## Configure and prepare

For Argon High full-document extraction:

```bash
rtk uv sync --locked
rtk uv run --locked python -m timeline_extraction.cli generate-config \
  --provider antigravity --goal consistency --output run.yaml
```

Choose a new config filename unless overwriting an existing one is intended. Substitute the requested provider and goal. Edit `models.<provider>` for the exact model ID, `reasoning_effort`, and `n_trails`. Account providers default to one trial; `cli_timeout` defaults to 600 positive seconds per process. `max_tokens` is not enforced by the CLI providers. Non-default sampling settings are not interchangeable with direct API inference.

Corpora are not bundled. Obtain the actual MATRES gold file and corresponding `.tml` documents; do not present synthetic smoke data as benchmark results. Gold has six tab-separated columns: `docid`, `verb1`, `verb2`, `eiid1`, `eiid2`, `relation`.

Replace the paths below with the available corpus paths:

```bash
rtk uv run --locked python -m timeline_extraction.cli prepare-data \
  --gold-file ./data/MATRES/platinum.txt \
  --docs-dir ./data/MATRES/raw/TBAQ-cleaned/te3-platinum \
  --data-name te3-platinum --config-file run.yaml --output-config run.yaml
```

This writes pair and multi JSON inputs and records their absolute paths. Existing prepared files require explicit `--overwrite`; do not overwrite unrelated inputs. YAML paths otherwise resolve relative to the working directory.

## Inspect, run, evaluate

```bash
rtk uv run --locked python -m timeline_extraction.cli research \
  --config-file run.yaml --plan
rtk uv run --locked python -m timeline_extraction.cli --tracking none research \
  --config-file run.yaml
```

- `--plan` is offline: no authentication, generation, or tracking SDK initialization. It does not prove that files exist or the account can access the model.
- `research` validates required inputs before generation and performs parsing, voting, and metrics automatically. Do not invent a separate evaluation command.
- `--model-name` overrides a model only when exactly one provider is selected. Repeat `--provider` and `--method` to compare backends or `zero-shot`/`few-shot` prompting.
- Use the root `--config` before the subcommand; command `--config-file` overrides it. Explicit CLI controls take precedence.
- Each CLI trial uses a fresh temporary workspace and restricted native agent execution. Native client/harness context differs from direct API inference. Antigravity can retain its own conversation history; this wrapper is not a hardened sandbox.

## Optional LangSmith tracing

Only enable tracing when sending corpus prompts and responses to the selected service is intended. Have the user set `LANGSMITH_API_KEY` privately in the environment; optionally set `LANGSMITH_PROJECT` and the appropriate `LANGSMITH_ENDPOINT`.

```bash
rtk uv run --locked --extra tracking python -m timeline_extraction.cli \
  --tracking langsmith research --config-file run.yaml
```

Tracking controls are root options, **before** `research`. Credentials alone do not enable tracing. Keep `--extra tracking` on traced `uv run` commands. This uploads experiment/generation traces and metrics; it does not automatically create a native LangSmith dataset Experiment or LLM-as-judge evaluator. Never claim it does.

If credentials and project are stored in `.env`, add `--env-file .env` to `uv run`, before `python`. The CLI does not load that file automatically. Never open or echo secret values merely to verify that the file exists.

If LangSmith returns `403 Forbidden`, check key permissions, the account's regional `LANGSMITH_ENDPOINT`, and `LANGSMITH_WORKSPACE_ID` for workspace-scoped access. Do not print the key or claim traces were uploaded: SDK upload warnings can coexist with successful local inference and metrics.

For an EU workspace (`eu.smith.langchain.com`), set `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com`; keys are not region-autodetected. Datasets live under **Datasets & Experiments**, not **Projects**. After fixing a tracing endpoint, a cached `--skip-model-eval` rerun can publish root metrics without fresh model requests; it does not recreate missing generation traces.

## Results, caching, and failures

- Find `research_summary.csv` under the configured results directory. Per-run paths include provider/model/mode/context/dataset/method/fingerprint.
- Read scores from `metrics.csv`, run metadata from `metrics.json`, voted predictions from `results.csv`, and generation evidence from `responses.jsonl` / `checkpoint.jsonl`. Report full-gold accuracy, coverage, F1, and temporal-cycle metrics only from actual artifacts.
- Rerunning identical settings and corpus contents reuses completed trials. Changed model/settings/input contents select a different fingerprint. Cached generations are not resent as fresh traces.
- To score saved responses without model credentials or fresh generation:

```bash
rtk uv run --locked python -m timeline_extraction.cli --tracking none research \
  --config-file run.yaml --skip-model-eval
```

Keep the original goal, methods, model, corpus, and generation settings to find the cache. Missing caches, authentication errors, quota limits, and incomplete native responses are failures, not valid predictions. Preserve unknown token counts as unknown. Report what executed, what was cached, and any missing prerequisite; do not claim a full corpus evaluation from a plan or small smoke test.

## Upload a prepared full-graph dataset

- Upload only when requested: prepared text and gold labels leave the machine.
- The research CLI has no dataset-upload subcommand. Use the installed LangSmith SDK (`Client`) with the same `uv run --extra tracking --env-file .env` environment.
- Build one example per prepared document. Use `inputs: {doc_id, text, relations}` from the prepared multi JSON and `outputs: {relations: [{event1, event2, relation}, ...]}` joined by document ID from the six-column MATRES gold file. The prepared `[RELATION]` placeholders are not reference labels.
- Check duplicate document IDs, missing gold documents, and the total gold relation count before uploading. Preserve the source files.
- Check for an existing named dataset and examples before `create_dataset` / `create_examples`; do not overwrite conflicting content. Stable example IDs make an interrupted upload resumable. Verify remote inputs and reference outputs after upload, not just the count.
- Keep an upload-ready JSON export locally if authorization blocks the request. A `403` is not a successful upload; correct `.env` permissions/region/workspace first, without sharing secrets in chat.
