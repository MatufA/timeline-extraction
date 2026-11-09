# Timeline Extraction: Temporal Relation Classification Research

**Research Repository for Temporal Relation Classification using Large Language Models**

This repository contains the implementation and evaluation framework for our research on temporal relation classification (TRC) using Large Language Models. The work investigates LLM prompting strategies for TRC and their effectiveness in assisting encoder models with cycle resolution.

## Abstract
Temporal relation classification (TRC) demands both accuracy and temporal consistency in event timeline extraction. Encoder based models achieve high accuracy but introduce inconsistencies because they rely on pairwise classification, while LLMs leverage global context to generate temporal graphs, improving consistency at the cost of accuracy. We assess LLM prompting strategies for TRC and their effectiveness in assisting encoder models with cycle resolution. Results show that while LLMs improve consistency, they struggle with accuracy and do not outperform a simple confidence-based cycle resolution approach.

## 🎯 Overview

This project implements a temporal relation classification system that:
- Extracts temporal relations between events in text using various LLM providers
- Supports multiple evaluation methodologies (zero-shot, few-shot)
- Provides comprehensive evaluation metrics and visualization tools
- Includes cycle detection and resolution mechanisms for temporal graphs
- Supports multiple temporal relation datasets (MATRES, TimeBank, AQUAINT, te3-platinum)

## 📊 Research Context

This system is designed for research in temporal relation classification and timeline extraction. It provides a standardized framework for evaluating LLM performance on temporal reasoning tasks, with particular focus on:

- **Temporal Relation Types**: BEFORE, AFTER, EQUAL, VAGUE
- **Evaluation Modes**: Pairwise relations, Multi-event relations, Combined approaches
- **Cycle Detection**: Automatic detection and resolution of temporal inconsistencies
- **Multi-Dataset Support**: Evaluation across multiple temporal relation datasets

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/MatufA/timeline-extraction.git
cd timeline-extraction

# Install the locked environment using uv
uv sync --locked
```

### Building distributions

```bash
rtk uv build
```

Package discovery includes only `timeline_extraction*`, so generated `results/` directories are not packaged. The optional `tracking` extra requires LangSmith `>=0.14.17,<1`; application tracing uses ingestion APIs unchanged by the SmithDB query migration.

### Basic Usage

```bash
# Supply MATRES gold labels and the corresponding TimeML documents first.
# API credentials and corpora are not bundled.
export OPENAI_API_KEY="your-openai-key"

# Select a research goal and provider
uv run python -m timeline_extraction.cli generate-config \
    --goal comparison --provider openai --output config.yaml

# Prepare both pairwise and document-level inputs; save their paths
uv run python -m timeline_extraction.cli prepare-data \
    --gold-file ./data/MATRES/platinum.txt \
    --docs-dir ./data/MATRES/raw/TBAQ-cleaned/te3-platinum \
    --config-file config.yaml --output-config config.yaml

# Inspect the effective plan without API calls, then run it
uv run python -m timeline_extraction.cli research --config-file config.yaml --plan
uv run python -m timeline_extraction.cli research --config-file config.yaml
```

For detailed usage examples, see `examples/cli_usage_examples.md` and `examples/elegant_pipeline_usage.py`.

### Coding-agent skill

The repository includes [timeline-extraction-cli](.agents/skills/timeline-extraction-cli/SKILL.md), a reusable skill for corpus preparation, account/API provider selection, graph evaluation, tracing, and cached reruns. Ask a compatible coding agent to “use the timeline-extraction-cli skill to run a document-level graph experiment.”

### Select by research goal

| Goal | Experiment | Question |
| --- | --- | --- |
| `classification` | Pairwise prompts and label parsing | How accurate are the relation predictions? |
| `consistency` | Document-level graph prompts and JSON/DOT parsing | How consistent are the predicted temporal graphs? |
| `comparison` | Both modes with the same provider, corpus, and methods | What is the accuracy/coverage/consistency tradeoff? |

Use repeated `--provider` and `--method` options to compare providers and zero-/few-shot prompting. `--model-name` overrides the model ID when exactly one provider is selected. These goals evaluate LLM relation predictions; they do not automatically reproduce encoder-assisted cycle-resolution experiments from the paper.

## Optional experiment tracing

Experiments are available through the CLI and Python API. Install the optional tracing SDKs:

```bash
uv sync --extra tracking
export LANGSMITH_API_KEY="your-langsmith-key"
uv run --extra tracking python -m timeline_extraction.cli --tracking langsmith research --config-file config.yaml

export LANGFUSE_PUBLIC_KEY="your-langfuse-public-key"
export LANGFUSE_SECRET_KEY="your-langfuse-secret-key"
uv run --extra tracking python -m timeline_extraction.cli --tracking langfuse research --config-file config.yaml

# Send to both configured backends, or disable an environment-selected backend.
uv run --extra tracking python -m timeline_extraction.cli --tracking both research --config-file config.yaml
TIMELINE_TRACKING=both uv run python -m timeline_extraction.cli --tracking none research --config-file config.yaml
```

Keep `--extra tracking` on enabled `uv run` invocations so optional dependencies remain selected. The integration uses [LangSmith custom instrumentation](https://docs.langchain.com/langsmith/annotate-code) and the [Langfuse SDK](https://langfuse.com/docs/observability/sdk/overview). The supported Langfuse SDK 5 requires Langfuse v4 for self-hosted targets; ensure your server is compatible before setting `--langfuse-base-url`.

Tracking options are root-level options: place them **before the subcommand**. CLI values override their environment equivalents:

| CLI option | Environment variable | Default |
| --- | --- | --- |
| `--tracking [none\|langsmith\|langfuse\|both]` | `TIMELINE_TRACKING` | `none` |
| `--langsmith-api-key` | `LANGSMITH_API_KEY` | Unset |
| `--langsmith-project` | `LANGSMITH_PROJECT` | `timeline-extraction` |
| `--langsmith-endpoint` | `LANGSMITH_ENDPOINT` | SDK default |
| `--langfuse-public-key` | `LANGFUSE_PUBLIC_KEY` | Unset |
| `--langfuse-secret-key` | `LANGFUSE_SECRET_KEY` | Unset |
| `--langfuse-base-url` | `LANGFUSE_BASE_URL` | SDK default |

Credentials alone never enable tracing. Explicit `--tracking none` overrides `TIMELINE_TRACKING`. Prefer environment variables for secrets to avoid shell-history exposure; tracing credentials are never stored in configuration, plans, or experiment artifacts.

Enabled backends receive a root trace with experiment settings (excluding credentials), metric rows, artifact paths, and failures, plus child LLM generations containing prompts, normalized responses, and aggregate token usage. **Prompts and responses may contain sensitive corpus text**: enable tracing only when you intend to send it to the configured services. Cached generations are not resent as new requests. Both backends may run simultaneously, and traces are flushed at the end of each experiment. Help, planning (`--plan`), data preparation, and configuration generation do not create tracking SDK clients.

Programmatic pipelines read the same environment variables by default. To override them explicitly, pass `tracking_config=TrackingConfig(backend="none")` (or configured backend credentials) to `TimelineExtractionPipeline`; import `TrackingConfig` from `timeline_extraction.tracking`.

Legacy local databases, run artifacts, dependencies, and generated UI build state are preserved and ignored; they are not a supported UI application.

## 📁 Repository Structure

```
timeline_extraction/
├── cli.py                    # Enhanced command-line interface
├── pipeline.py               # Elegant evaluation pipeline
├── config.py                 # Configuration management
├── tracking.py               # Optional LangSmith/Langfuse experiment tracing
├── models/                   # LLM provider implementations
│   ├── LLModel.py           # Base model interface
│   ├── OpenAIClient.py      # OpenAI API client
│   ├── HuggingFaceClient.py # HuggingFace models
│   ├── gemini.py            # Google Gemini client
│   ├── _responses.py       # Shared chat response normalization
│   ├── llama3.py           # Groq provider adapter
│   └── TogetherAIClient.py  # Together AI client
├── data/                     # Data processing modules
│   ├── preprocessing.py      # Data preprocessing
│   ├── postprocessing.py     # Result postprocessing
│   ├── tml_parser.py        # TML file parsing
│   └── count_graph_links.py # Graph analysis utilities
├── prompts/                  # Prompt templates
│   └── Prompt.py            # Prompt classes
├── visualization/            # Visualization tools
│   ├── data.py              # Data visualization
│   └── graph.py             # Graph visualization
├── examples/                 # Usage examples
│   ├── cli_usage_examples.md # CLI usage documentation
│   └── elegant_pipeline_usage.py # Programmatic usage examples
├── metrics.py               # Evaluation metrics
├── graph.py                 # Temporal graph operations
├── graph_cycles_breaker.py  # Cycle detection and resolution
├── tokens_counter.py        # Token counting utilities
└── utils.py                 # General utilities
```

## 🔧 Configuration

The system uses YAML configuration files for managing experiments. Generate a default configuration:

```bash
uv run python -m timeline_extraction.cli generate-config --output config.yaml
```

Alternatively, set API keys via environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-google-key"  # GOOGLE_API_KEY is also supported
export TOGETHER_API_KEY="your-together-key"
export GROQ_API_KEY="your-groq-key"
```

### Supported LLM Providers

| Provider | Default model | Current API/SDK |
| --- | --- | --- |
| OpenAI | `gpt-6-luna` | Responses API; Chat Completions for explicit compatible endpoints |
| Google | `gemini-3.8-flash` | `google-genai`, replacing the retired `google-generativeai` SDK |
| Together | `openai/gpt-oss-120b` | Together Chat Completions |
| Groq | `openai/gpt-oss-20b` | Groq Chat Completions |
| HuggingFace | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | Local Transformers generation, CPU by default |
| Antigravity (`antigravity`) | `gemini-4-argon-high` | Official `agy` CLI with cached Google account login |
| Codex (`codex`) | `gpt-5.6-sol` | Official `codex exec` CLI with cached ChatGPT login |
| Claude Code (`claude`) | `sonnet` | Official `claude` CLI with cached claude.ai subscription login |

Defaults follow the official [OpenAI](https://developers.openai.com/api/docs/guides/latest-model), [Gemini](https://ai.google.dev/gemini-api/docs/models), [Together](https://docs.together.ai/docs/serverless/models), [Groq](https://console.groq.com/docs/models), and [Transformers](https://huggingface.co/docs/transformers/en/conversations) documentation. Availability depends on the provider account. Any supported model ID can be configured without editing the factory.

Provider settings include `max_tokens` (hosted default: 8192, including reasoning tokens), `temperature`, `n_trails` (default: 5), and optional `reasoning_effort`. Trials use separate requests and a strict majority vote. OpenAI reasoning models omit unsupported sampling parameters; an explicit supported `none` effort preserves configured temperature. Use `models.huggingface.device` for CUDA/MPS, and `HF_TOKEN` for gated models.

Global `--config` settings are merged with command-specific `--config-file` settings; explicit CLI controls take precedence. Credentials resolve from configuration or environment and are excluded from plans and experiment metadata. Dataset overrides support `gold_file` and `prepared_files: {pair: ..., multi: ...}` with paths relative to the working directory. `prepare-data --output-config` writes absolute corpus paths.

### Account-backed CLI providers

Install the official [Antigravity CLI](https://antigravity.google/docs/getting-started?tab=cli), [Codex CLI](https://developers.openai.com/codex/cli), or [Claude Code](https://code.claude.com/docs/en/setup). Keep `agy`, `codex`, or `claude` on `PATH`, and authenticate using the CLI's native flow:

```bash
rtk agy                # Sign in with your Google account, then exit.
rtk codex login        # Select ChatGPT account login.
rtk claude auth login  # Select claude.ai subscription login, not --console.
```

Generate a configuration for the selected provider; Antigravity defaults to Argon High:

```bash
rtk uv run --locked python -m timeline_extraction.cli generate-config \
    --provider antigravity --goal consistency --output antigravity.yaml
```

Prepare your corpus using that configuration as described in [CLI usage](examples/cli_usage_examples.md#configure-prepare-inspect-run), then run:

```bash
rtk uv run --locked python -m timeline_extraction.cli research \
    --config-file antigravity.yaml --plan
rtk uv run --locked --extra tracking python -m timeline_extraction.cli \
    --tracking langsmith research --config-file antigravity.yaml
```

Set `LANGSMITH_API_KEY` in the environment before the traced command, or use `--tracking none` without `--extra tracking` for local-only metrics. `consistency` performs document-level graph extraction and evaluation; `comparison` also runs pairwise evaluation. Substitute `--provider codex` or `--provider claude` when generating configuration for those accounts. Account entitlement, model availability, quotas, and provider terms still apply.

If the key and project are in a local `.env`, add `--env-file .env` to `uv run` (for example, `rtk uv run --locked --extra tracking --env-file .env python -m timeline_extraction.cli --tracking langsmith research --config-file antigravity.yaml`). The CLI does not automatically load `.env`; keep it untracked.

EU-hosted workspaces require `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com`; the default US API rejects EU keys. See [regional endpoint configuration](https://kb.langchain.com/articles/6271700631-pointing-the-langsmith-sdk-and-cli-at-a-non-us-regional-endpoint). Datasets appear under **Datasets & Experiments**; tracing projects appear under **Projects**.

CLI providers default to one trial. Configure `n_trails`, `reasoning_effort`, and `cli_timeout` (positive seconds, default 600) under `models.<provider>`. Select exact Antigravity model slugs using `rtk agy models`. Each trial starts a fresh process in a temporary workspace, supplies the conversation over stdin, and uses restricted tools rather than permission-bypass flags. Codex runs read-only and rejects any surfaced tool activity; this is not a general-purpose security sandbox. CLI harness instructions and overhead differ from direct API inference. Antigravity retains its own native conversation history.

Account providers reject `api_key`, `base_url`, and nonzero `temperature`; API-auth environment overrides are removed from their child processes. They do not extract credentials or fall back to API billing. `max_tokens` is an API-provider control, not an enforced CLI output limit; CLI generation limits remain controlled by the native client. Failed, interrupted, or incomplete runs fail the experiment rather than being scored as labels. Missing token usage stays unknown.

LangChain's standard provider integrations use API credentials. Its [Codex OAuth adapter](https://reference.langchain.com/python/langchain-openai/chat_models/codex) is explicitly experimental and unofficial, so account-backed providers use the official CLI protocols rather than private token integrations.

## 📈 Evaluation Pipeline

### Supported Methods

1. **Zero-shot**: Direct evaluation without examples
2. **Few-shot**: Evaluation with in-context examples

### Supported Modes

1. **Pair**: Pairwise temporal relation classification
2. **Multi**: Multi-event temporal relation extraction
3. **Comb**: Pairwise classification over prepared contextual event pairs

### Graph evaluation

Document-level (`multi`) evaluation reports separate **explicit** and **expanded** scores. Expanded scoring completes the voted graph transitively: A BEFORE B and B BEFORE C imply A BEFORE C. Equality components propagate equality and ordering; valid explicit labels are preserved.

Completion never turns disconnected pairs into VAGUE or infers through contradictory graph components. `expanded_results.csv` marks inferred edges with `is_inferred`; `results.csv` remains the original voted output. See [output formats and cached rescoring](examples/cli_usage_examples.md#outputs-and-resume).

Expanded scores measure graph entailment, not additional model generations. Compare the same scoring view when assessing model changes; historical scores were explicit-only.

### Datasets

- **MATRES**: Multi-Aspect Temporal Relation Extraction [repo](https://github.com/qiangning/MATRES)
- **OmniTemp**: Native JSON from [GlobalZeroShotTRE's dataset source](https://github.com/AlonEirew/event-relation-resources/tree/main/tre_datasets/OmniTemp), prepared with `prepare-data --dataset-format omnitemp --input-dir <split-directory>`. See the [OmniTemp CLI example](examples/cli_usage_examples.md#omnitemp).
- **Narrative Time**: First timeline-based annotation framework that achieves full coverage of all possible TLINKS [repo](https://github.com/text-machine-lab/nt)

OmniTemp uses the existing BEFORE/AFTER/EQUAL/VAGUE labels. Its [usage agreement](https://github.com/AlonEirew/event-relation-resources/blob/main/tre_datasets/OmniTemp/LICENSE.txt) permits non-commercial research and educational use only; corpus files are not bundled with this package.

## 🧪 Research Workflow

1. **Setup**: Select a research goal/provider, set its API key, and supply the corpus
2. **Preparation**: Run `prepare-data` against MATRES/TimeML inputs or a native OmniTemp split
3. **Plan**: Inspect `research --plan` before making paid requests
4. **Evaluation**: Run `research`, or use the explicit single/batch commands
5. **Analysis**: Compare metrics and the provider/mode/method rows in `research_summary.csv`

For detailed workflow examples, see `examples/cli_usage_examples.md`.

## 📊 Results and Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-relation-type metrics (BEFORE, AFTER, EQUAL, VAGUE)
- **Cycle Detection**: Temporal consistency analysis
- **Error Analysis**: Detailed error categorization
- **Token Usage**: Per-trial provider token usage

Outputs contain normalized/raw responses, voted relations, metrics, and metadata. They are isolated under `results/<provider>/<model>/<mode>/<context>/<dataset>/<method>/<fingerprint>/`; the fingerprint includes generation settings and input contents. Missing or invalid predictions remain in the gold denominator, and `coverage` reports the predicted fraction. Cycle reporting counts documents containing a contradiction, including equality-induced contradictions.

Re-running the same experiment resumes completed trials without repeating requests. `--skip-model-eval` recomputes results from cached responses without constructing a provider client or requiring credentials; retain the same inputs and generation settings to identify the cache.

## 🔬 Research Features

- **Cycle Detection and Resolution**: Sophisticated algorithms for temporal consistency
- **Multi-Model Comparison**: Standardized framework for LLM evaluation
- **Visualization Tools**: Built-in tools for temporal graphs and relation patterns
- **Elegant Pipeline**: Clean, modular architecture for temporal relation extraction

## 📚 Usage

The system provides both CLI and programmatic interfaces:

- **CLI Commands**: `research`, `prepare-data`, `model evaluate`, `model batch-evaluate`, `demo-pipeline`, `generate-config`, `analyze-data`
- **Programmatic API**: `build_research_plan`, `TimelineExtractionPipeline`, and `BatchExperimentRunner`; see `examples/elegant_pipeline_usage.py`

For comprehensive usage documentation, see `examples/cli_usage_examples.md`.

## 🔬 Research Reproducibility

This repository is designed for research reproducibility:

1. **Setup Environment**: Follow the installation instructions above
2. **Configure API Keys**: Set up your API keys in the configuration file
3. **Run Experiments**: Use the provided CLI commands or example scripts
4. **Analyze Results**: Review the generated CSV files and metrics

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📖 Citation

If you use this system in your research, please cite:
```
@inproceedings{meir-bar-2025-llms,
    title = "Can {LLM}s Help Encoder Models Maintain Both High Accuracy and Consistency in Temporal Relation Classification?",
    author = "Meir, Adiel  and
      Bar, Kfir",
    editor = "Flek, Lucie  and
      Narayan, Shashi  and
      Phương, L{\^e} Hồng  and
      Pei, Jiahuan",
    booktitle = "Proceedings of the 18th International Natural Language Generation Conference",
    month = oct,
    year = "2025",
    address = "Hanoi, Vietnam",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.inlg-main.41/",
    pages = "716--733",
    abstract = "Temporal relation classification (TRC) demands both accuracy and temporal consistency in event timeline extraction. Encoder-based models achieve high accuracy but introduce inconsistencies because they rely on pairwise classification, while LLMs leverage global context to generate temporal graphs, improving consistency at the cost of accuracy. We assess LLM prompting strategies for TRC and their effectiveness in assisting encoder models with cycle resolution. Results show that while LLMs improve consistency, they struggle with accuracy and do not outperform a simple confidence-based cycle resolution approach. Our code is publicly available at: \url{https://github.com/MatufA/timeline-extraction}."
}
```

## 🆘 Support

For questions about this research:
- Open an issue on GitHub for technical questions
- Check the examples in the `examples/` directory
- Review the CLI usage documentation in `examples/cli_usage_examples.md`
