# Research CLI usage

## Configure, prepare, inspect, run

Install the locked environment and generate a goal-based configuration:

```bash
uv sync --locked
export OPENAI_API_KEY="your-key"
uv run python -m timeline_extraction.cli generate-config \
    --goal comparison --provider openai --output config.yaml
```

Obtain [MATRES gold labels](https://github.com/qiangning/MATRES) and the corresponding TimeML documents. Corpora and credentials are not bundled. Prepare both experimental modes and save their paths:

```bash
uv run python -m timeline_extraction.cli prepare-data \
    --gold-file ./data/MATRES/platinum.txt \
    --docs-dir ./data/MATRES/raw/TBAQ-cleaned/te3-platinum \
    --data-name te3-platinum \
    --config-file config.yaml --output-config config.yaml

uv run python -m timeline_extraction.cli research --config-file config.yaml --plan
uv run python -m timeline_extraction.cli research --config-file config.yaml
```

`prepare-data` consumes a six-column tab-separated gold file (`docid`, `verb1`, `verb2`, `eiid1`, `eiid2`, `relation`) and matching `.tml` documents. It writes JSON arrays despite the historical serializer name `prepare_as_jsonl`. Existing prepared files require `--overwrite` to replace. `--output-config` records absolute prepared/gold paths and selects the corpus. Without that option, use the printed paths in configuration or the standard directory layout.

The default research goal is `classification` (pairwise accuracy). `consistency` runs document-level temporal-graph extraction. `comparison` runs both modes under the same model/method/corpus controls. These goals evaluate LLM predictions, not the paper's entire encoder-assisted cycle-resolution workflow.

## Compare providers and prompting methods

```bash
export GEMINI_API_KEY="your-google-key"
uv run python -m timeline_extraction.cli research \
    --config-file config.yaml --goal comparison \
    --provider openai --provider google \
    --method zero-shot --method few-shot
```

Use `google` as the provider name for Gemini. Other provider names are `together`, `groq`, and `huggingface`. Set `TOGETHER_API_KEY`, `GROQ_API_KEY`, or `HF_TOKEN` as needed. `GOOGLE_API_KEY` is also supported. Model defaults and official provider references are listed in the [README](../README.md#supported-llm-providers).

Override a model ID without changing code:

```bash
uv run python -m timeline_extraction.cli research \
    --config-file config.yaml --goal classification \
    --provider openai --model-name gpt-6.1-sol --plan
```

A model override requires exactly one provider. Provider routing is explicit, so a model such as `openai/gpt-oss-20b` can correctly run on Groq rather than being mistaken for an OpenAI or local model.

## Configuration controls

```yaml
research:
  goal: comparison
  providers: [openai, google]
  methods: [zero-shot, few-shot]
  dataset: custom
models:
  openai:
    default_model: gpt-6-luna
    api_key: null
    max_tokens: 8192
    temperature: 0.0
    n_trails: 5
  google:
    default_model: gemini-3.8-flash
    api_key: null
    max_tokens: 8192
    temperature: 0.0
    n_trails: 5
evaluation:
  use_vague: true
  overwrite: false
  skip_model_eval: false
data:
  data_path: ./data
  results_path: ./results
  datasets:
    custom:
      gold_file: ./corpus/gold.tsv
      prepared_files:
        pair: ./corpus/pair.json
        multi: ./corpus/multi.json
```

Paths supplied in YAML are relative to the working directory unless absolute. Without overrides, the standard paths are:

- Gold: `data/MATRES/platinum.txt`, `timebank.txt`, or `aquaint.txt`.
- Prepared: `data/TRC/raw_text/{pair,multi}_<dataset>_text_w_relations_prepared.json`.
- Comb inputs: `data/TRC/raw_text/comb_<dataset>_{full_context,minimal_context}_text_w_relations_prepared.json`.

A `prepared_file` override can select one input for a single mode; use `prepared_files` for comparisons. Local models default to CPU; configure `models.huggingface.device` as `cuda:0` or `mps` when available.

`n_trails` controls independent generations and strict-majority voting. Invalid or conflicting predictions abstain. Reasoning tokens consume the output budget; increase `max_tokens` if the provider returns an incomplete response. OpenAI reasoning models omit unsupported `temperature`. Provider refusals/truncation fail the experiment rather than becoming predictions.

A global `--config` file is merged with the command's `--config-file`; command-specific configuration overrides matching nested settings, and explicit CLI controls take precedence. Plans and metadata exclude API keys.

## Opt-in experiment tracing

Install the SDKs with `uv sync --extra tracking`, and keep `--extra tracking` on enabled `uv run` invocations so optional dependencies remain selected. All tracking controls belong before the subcommand:

```bash
export LANGSMITH_API_KEY="your-langsmith-key"
export LANGFUSE_PUBLIC_KEY="your-langfuse-public-key"
export LANGFUSE_SECRET_KEY="your-langfuse-secret-key"

# CLI-selected backend; use langsmith or langfuse for only one service.
uv run --extra tracking python -m timeline_extraction.cli --tracking both \
    --langsmith-project timeline-extraction research --config-file config.yaml

# Environment-selected backend, with an explicit CLI override to disable it.
export TIMELINE_TRACKING=langsmith
uv run --extra tracking python -m timeline_extraction.cli research --config-file config.yaml
uv run python -m timeline_extraction.cli --tracking none research --config-file config.yaml
```

`--tracking` accepts `none` (default), `langsmith`, `langfuse`, or `both`; credentials alone do not enable tracing. CLI values override environment variables. Root options `--langsmith-api-key`, `--langsmith-project` (default `timeline-extraction`), `--langsmith-endpoint`, `--langfuse-public-key`, `--langfuse-secret-key`, and `--langfuse-base-url` map respectively to `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `LANGSMITH_ENDPOINT`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL`. Omitted endpoints use SDK defaults.

Use environment variables for secrets rather than command history. Credentials are excluded from configuration, plans, and artifacts. Enabled tracing sends experiment settings, metrics, artifact paths, failures, and child generation prompts/responses with aggregate token usage to the selected services; corpus text may therefore leave your machine. Cached generations are not resent as fresh requests. Traces flush after each experiment. Help, `--plan`, `prepare-data`, and `generate-config` do not initialize tracing clients. See the [tracking reference](../README.md#optional-experiment-tracing) for the complete option table.

Backend references: [LangSmith custom instrumentation](https://docs.langchain.com/langsmith/annotate-code) and [Langfuse SDK overview](https://langfuse.com/docs/observability/sdk/overview). The supported Langfuse SDK 5 requires Langfuse v4 for self-hosted targets; use a compatible server when configuring `LANGFUSE_BASE_URL`.

## Explicit single and batch evaluation

```bash
uv run python -m timeline_extraction.cli model evaluate \
    --provider openai --model-name gpt-6-luna \
    --method zero-shot --mode pair --data-name te3-platinum \
    --config-file config.yaml

uv run python -m timeline_extraction.cli model batch-evaluate \
    --provider openai \
    --model-names gpt-6-luna --model-names gpt-6.1-sol \
    --methods zero-shot --methods few-shot \
    --mode multi --data-name te3-platinum --config-file config.yaml
```

Pair/comb modes infer label parsing; multi mode infers JSON/DOT relation parsing. `--parser-type` selects an explicit alternate. `--full-context/--minimal-context` selects the comb input variant. `--use-vague/--no-use-vague`, `--overwrite/--no-overwrite`, and `--output-dir` override single/batch evaluation controls.

`demo-pipeline` is a real paid batch evaluation against your prepared corpus, not an offline demonstration.

## Outputs and resume

Each experiment writes `responses.jsonl`, `checkpoint.jsonl`, `parsed.csv`, `results.csv`, `metrics.csv`, and `metrics.json` beneath:

```text
results/<provider>/<model>/<mode>/<context>/<dataset>/<method>/<fingerprint>/
```

The fingerprint includes experimental/generation settings and prepared/gold contents. API keys are excluded, so credential rotation does not invalidate a cache. Re-running identical inputs resumes completed trials. Changing corpus contents or model settings selects separate artifacts. Pre-upgrade artifacts are not reused automatically.

`research_summary.csv` identifies provider, model, goal, mode, method, success/error, and artifact paths. Metrics report full-gold accuracy, coverage, per-relation precision/recall/F1, macro/weighted/micro F1, and documents containing temporal cycles. Equality components are contracted before consistency checks. Missing predictions lower accuracy and coverage rather than disappearing from evaluation.

Re-evaluate saved responses without provider credentials:

```bash
uv run python -m timeline_extraction.cli research \
    --config-file config.yaml --skip-model-eval
```

Keep the same inputs, provider/model settings, goal, and methods so the cached response paths match. Missing inputs are checked before the first research request. Failures produce a nonzero exit code and recorded error.

## Data analysis and checks

```bash
uv run python -m timeline_extraction.cli analyze-data \
    --data-path ./data/MATRES/platinum.txt --output analysis.json
uv run pytest --override-ini addopts='' -q
```

Use `uv run python -m timeline_extraction.cli --verbose research ...` for debug logging. For programmatic goal selection, see [elegant_pipeline_usage.py](elegant_pipeline_usage.py).
