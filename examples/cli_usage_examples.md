# CLI Usage Examples - Elegant Pipeline

This document provides examples of how to use the updated CLI with the new elegant pipeline implementation.

## 🚀 Quick Start

### 1. Generate Configuration File

First, generate a default configuration file:

```bash
python -m timeline_extraction.cli generate-config --output config.yaml
```

### 2. Run Single Model Evaluation

Evaluate a single model using the new pipeline:

```bash
python -m timeline_extraction.cli model evaluate \
    --model-name "gpt-4o-mini" \
    --method "zero-shot" \
    --mode "comb" \
    --data-name "te3-platinum" \
    --output-dir "./results/single" \
    --config-file "config.yaml"
```

### 3. Run Batch Evaluation

Evaluate multiple models and methods:

```bash
python -m timeline_extraction.cli model batch-evaluate \
    --model-names "gpt-4o-mini" \
    --model-names "gpt-4o" \
    --methods "zero-shot" \
    --methods "few-shot" \
    --mode "comb" \
    --data-name "te3-platinum" \
    --output-dir "./results/batch" \
    --config-file "config.yaml"
```

### 4. Demo Pipeline

Run a demonstration of the new elegant pipeline:

```bash
python -m timeline_extraction.cli demo-pipeline \
    --model-names "gpt-4o-mini" \
    --methods "zero-shot" \
    --methods "few-shot" \
    --mode "comb" \
    --data-name "te3-platinum" \
    --output-dir "./results/demo"
```

### 5. Analyze Data

Analyze dataset characteristics:

```bash
python -m timeline_extraction.cli analyze-data \
    --data-path "./data/te3-platinum.json" \
    --output "./analysis_results.json"
```

## 📋 Command Reference

### Main Commands

- `model evaluate`: Run single model evaluation
- `model batch-evaluate`: Run batch evaluations across multiple models
- `demo-pipeline`: Demonstrate the new elegant pipeline
- `generate-config`: Generate default configuration file
- `analyze-data`: Analyze dataset characteristics

### Common Options

- `--model-name`: Model to use (e.g., "gpt-4o-mini", "gemini-1.5-pro")
- `--model-names`: Multiple model names (for batch operations)
- `--method`: Evaluation method ("zero-shot" or "few-shot")
- `--methods`: Multiple evaluation methods (for batch operations)
- `--mode`: Evaluation mode ("pair", "multi", or "comb")
- `--data-name`: Dataset name ("te3-platinum", "timebank", "aquaint")
- `--use-vague/--no-use-vague`: Include VAGUE relations (default: True)
- `--parser-type`: Parser type ("json" or "label", default: "json")
- `--overwrite/--no-overwrite`: Overwrite existing results (default: False)
- `--skip-model-eval/--no-skip-model-eval`: Skip model evaluation (default: False)
- `--full-context/--no-full-context`: Use full context (default: True)
- `--output-dir`: Output directory for results
- `--config-file`: Configuration file path
- `--gpu-device`: GPU device to use (default: 0)
- `--verbose/-v`: Enable verbose logging
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)

## 🔧 Configuration

### Configuration File Structure

```yaml
models:
  openai:
    api_key: "your-openai-key"
    default_model: "gpt-4o-mini"
    max_tokens: 2000
    temperature: 0.0
  
  google:
    api_key: "your-google-key"
    default_model: "gemini-1.5-pro"

evaluation:
  default_method: "zero-shot"
  default_mode: "comb"
  use_vague: true
  parser_type: "json"
  overwrite: false

data:
  data_path: "./data"
  results_path: "./results"
  cache_path: "./cache"
```

### Environment Variables

Set API keys via environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"
```

## 📊 Output Structure

The CLI generates the following output structure:

```
results/
├── final_metrics/
│   ├── zero-shot/
│   │   └── comb-te3-platinum-gpt-4o-mini-zero-shot-w_vague-completion-results.csv
│   └── few-shot/
│       └── comb-te3-platinum-gpt-4o-mini-few-shot-w_vague-completion-results.csv
├── batch_summary.csv
└── demo_summary.csv
```

## 🎯 Advanced Usage

### Custom Configuration

Create a custom configuration file:

```yaml
models:
  openai:
    api_key: "your-key"
    default_model: "gpt-4o"
    max_tokens: 4000
    temperature: 0.1

evaluation:
  default_method: "few-shot"
  default_mode: "multi"
  use_vague: false
  parser_type: "json"
  overwrite: true

data:
  data_path: "/path/to/your/data"
  results_path: "/path/to/results"
```

### Verbose Logging

Enable verbose logging for debugging:

```bash
python -m timeline_extraction.cli --verbose model evaluate \
    --model-name "gpt-4o-mini" \
    --method "zero-shot" \
    --mode "comb" \
    --data-name "te3-platinum"
```

### Multiple GPU Support

Use different GPU devices:

```bash
python -m timeline_extraction.cli model batch-evaluate \
    --model-names "gpt-4o-mini" \
    --model-names "gpt-4o" \
    --methods "zero-shot" \
    --methods "few-shot" \
    --gpu-device 1 \
    --mode "comb" \
    --data-name "te3-platinum"
```

## 🔍 Troubleshooting

### Common Issues

1. **API Key Not Found**: Ensure API keys are set in environment variables or config file
2. **Model Not Found**: Check model name spelling and availability
3. **Data Path Not Found**: Verify data directory structure
4. **GPU Memory Issues**: Reduce batch size or use smaller models

### Debug Mode

Enable debug logging:

```bash
python -m timeline_extraction.cli --log-level DEBUG model evaluate \
    --model-name "gpt-4o-mini" \
    --method "zero-shot" \
    --mode "comb" \
    --data-name "te3-platinum"
```

## 📈 Performance Tips

1. **Use Batch Evaluation**: More efficient for multiple models/methods
2. **Skip Model Eval**: Use `--skip-model-eval` to process existing results
3. **GPU Memory**: Monitor GPU usage and adjust batch sizes
4. **Configuration**: Use config files for consistent settings

## 🔄 Migration from Old CLI

The new CLI maintains backward compatibility while providing enhanced features:

- All existing command-line options work
- New pipeline architecture provides better error handling
- Enhanced logging and progress tracking
- Improved configuration management
- Better resource cleanup

## 📚 Examples

### Complete Workflow

```bash
# 1. Generate config
python -m timeline_extraction.cli generate-config --output my_config.yaml

# 2. Edit config file with your API keys
# 3. Run batch evaluation
python -m timeline_extraction.cli model batch-evaluate \
    --model-names "gpt-4o-mini" \
    --model-names "gpt-4o" \
    --methods "zero-shot" \
    --methods "few-shot" \
    --mode "comb" \
    --data-name "te3-platinum" \
    --output-dir "./results/experiment_1" \
    --config-file "my_config.yaml"

# 4. Check results
ls -la ./results/experiment_1/
```

### Quick Demo

```bash
# Run a quick demonstration
python -m timeline_extraction.cli demo-pipeline \
    --model-names "gpt-4o-mini" \
    --methods "zero-shot" \
    --mode "comb" \
    --data-name "te3-platinum" \
    --output-dir "./results/quick_demo"
```
