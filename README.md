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

# Install dependencies using uv (recommended) or pip
uv sync
# OR
pip install -e .

```

### Basic Usage

```bash
# Generate configuration file
python -m timeline_extraction.cli generate-config --output config.yaml

# Run single model evaluation
python -m timeline_extraction.cli model evaluate \
    --model-name "gpt-4o-mini" \
    --method "zero-shot" \
    --mode "comb" \
    --data-name "te3-platinum" \
    --config-file "config.yaml"
```

For detailed usage examples, see `examples/cli_usage_examples.md` and `examples/elegant_pipeline_usage.py`.

## 📁 Repository Structure

```
timeline_extraction/
├── cli.py                    # Enhanced command-line interface
├── pipeline.py               # Elegant evaluation pipeline
├── config.py                 # Configuration management
├── models/                   # LLM provider implementations
│   ├── LLModel.py           # Base model interface
│   ├── OpenAIClient.py      # OpenAI API client
│   ├── HuggingFaceClient.py # HuggingFace models
│   ├── gemini.py            # Google Gemini client
│   ├── llama.py             # Llama models
│   ├── llama3.py            # Llama 3 models
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
python -m timeline_extraction.cli generate-config --output config.yaml
```

Alternatively, set API keys via environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"
export TOGETHER_API_KEY="your-together-key"
export GROQ_API_KEY="your-groq-key"
```

### Supported LLM Providers

- **OpenAI**: GPT-4o, GPT-4o-mini
- **Google**: Gemini-1.5-pro, Gemini-1.5-flash
- **HuggingFace**: Local and hosted models (Llama etc.)
- **Together AI**: Various open-source models
- **Groq**: Fast inference models

## 📈 Evaluation Pipeline

### Supported Methods

1. **Zero-shot**: Direct evaluation without examples
2. **Few-shot**: Evaluation with in-context examples

### Supported Modes

1. **Pair**: Pairwise temporal relation classification
2. **Multi**: Multi-event temporal relation extraction
3. **Comb**: Combined approach

### Datasets

- **MATRES**: Multi-Aspect Temporal Relation Extraction [repo](https://github.com/qiangning/MATRES)
- **Narrative Time**: First timeline-based annotation framework that achieves full coverage of all possible TLINKS [repo](https://github.com/text-machine-lab/nt)

## 🧪 Research Workflow

1. **Setup**: Generate configuration file and set API keys
2. **Data Analysis**: Analyze dataset characteristics
3. **Single Evaluation**: Run single model evaluation
4. **Batch Evaluation**: Run multiple models/methods
5. **Demo Pipeline**: Complete demonstration

For detailed workflow examples, see `examples/cli_usage_examples.md`.

## 📊 Results and Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-relation-type metrics (BEFORE, AFTER, EQUAL, VAGUE)
- **Cycle Detection**: Temporal consistency analysis
- **Error Analysis**: Detailed error categorization
- **Token Usage**: Cost and efficiency metrics

Results are saved in CSV format with detailed breakdowns by relation type and dataset.

## 🔬 Research Features

- **Cycle Detection and Resolution**: Sophisticated algorithms for temporal consistency
- **Multi-Model Comparison**: Standardized framework for LLM evaluation
- **Visualization Tools**: Built-in tools for temporal graphs and relation patterns
- **Elegant Pipeline**: Clean, modular architecture for temporal relation extraction

## 📚 Usage

The system provides both CLI and programmatic interfaces:

- **CLI Commands**: `model evaluate`, `model batch-evaluate`, `demo-pipeline`, `generate-config`, `analyze-data`
- **Programmatic API**: See `examples/elegant_pipeline_usage.py` for detailed examples

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
