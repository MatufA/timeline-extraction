"""Command-line entry points for reproducible temporal-relation research."""

import json
import logging
from pathlib import Path
from typing import Optional

import click
import yaml

from timeline_extraction.config import ConfigManager, create_default_config
from timeline_extraction.data.omnitemp import prepare_omnitemp
from timeline_extraction.data.preprocessing import load_data, generate_training_dataset, prepare_as_jsonl
from timeline_extraction.pipeline import (
    BatchExperimentRunner,
    PipelineExperimentConfig,
    TimelineExtractionPipeline,
)
from timeline_extraction.research import build_research_plan, describe_research_plan
from timeline_extraction.exceptions import DataError
from timeline_extraction.tracking import TrackingConfig

PROVIDERS = ["openai", "google", "together", "groq", "huggingface", "antigravity", "codex", "claude"]
GOALS = ["classification", "consistency", "comparison"]


def setup_config_manager(config_file: Optional[str], ctx_config: dict) -> ConfigManager:
    manager = ConfigManager()
    if ctx_config:
        manager.merge_config(ctx_config)
    if config_file:
        # A command-specific file overrides the global file section by section.
        with open(config_file) as file:
            settings = yaml.safe_load(file) if Path(config_file).suffix in {".yaml", ".yml"} else json.load(file)
        manager.merge_config(settings)
    return manager


def _pipeline(manager: ConfigManager, ctx) -> TimelineExtractionPipeline:
    return TimelineExtractionPipeline(config_manager=manager, tracking_config=ctx.obj["tracking"])


def _report(result):
    if not result.success:
        raise click.ClickException(result.error)
    click.echo(f"Results: {result.results_path}")
    if result.expanded_results_path is not None:
        click.echo(f"Expanded results: {result.expanded_results_path}")
    click.echo(f"Metrics: {result.metrics_path}")
    click.echo(f"Metadata: {result.metadata_path}")


def add_common_evaluation_options(func):
    options = [
        click.option("--mode", type=click.Choice(["pair", "multi", "comb"]), required=True),
        click.option("--data-name", required=True, help="Dataset key in configuration."),
        click.option("--provider", type=click.Choice(PROVIDERS), help="Explicit provider; required for ambiguous model IDs."),
        click.option("--gpu-device", type=int, default=None, help="Override local model device with a CUDA index."),
        click.option("--use-vague/--no-use-vague", default=None),
        click.option("--parser-type", type=click.Choice(["label", "json"]), default=None),
        click.option("--overwrite/--no-overwrite", default=None),
        click.option("--skip-model-eval/--no-skip-model-eval", default=None),
        click.option("--output-dir", type=click.Path(), default=None),
        click.option("--full-context/--minimal-context", default=True),
        click.option("--config-file", type=click.Path(exists=True)),
    ]
    for option in options:
        func = option(func)
    return func


def _experiment(manager, model_name, method, **options):
    evaluation = manager.evaluation_config
    for name in ("use_vague", "overwrite", "skip_model_eval"):
        if options[name] is None:
            options[name] = getattr(evaluation, name)
    options["output_dir"] = options["output_dir"] or manager.data_config.results_path
    options["gpu_device"] = options["gpu_device"] if options["gpu_device"] is not None else 0
    options["parser_type"] = options["parser_type"] or evaluation.parser_type
    return PipelineExperimentConfig(model_name=model_name, method=method, **options)


@click.group()
@click.option("--config", type=click.Path(exists=True), help="Global YAML/JSON configuration.")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), default="INFO")
@click.option("--tracking", type=click.Choice(["none", "langsmith", "langfuse", "both"]), envvar="TIMELINE_TRACKING", default="none", show_default=True)
@click.option("--langsmith-api-key", envvar="LANGSMITH_API_KEY", help="Prefer environment credentials to avoid shell history.")
@click.option("--langsmith-project", envvar="LANGSMITH_PROJECT", default="timeline-extraction", show_default=True)
@click.option("--langsmith-endpoint", envvar="LANGSMITH_ENDPOINT")
@click.option("--langfuse-public-key", envvar="LANGFUSE_PUBLIC_KEY")
@click.option("--langfuse-secret-key", envvar="LANGFUSE_SECRET_KEY", help="Prefer environment credentials to avoid shell history.")
@click.option("--langfuse-base-url", envvar="LANGFUSE_BASE_URL")
@click.pass_context
def cli(ctx, config, verbose, log_level, tracking, **tracking_options):
    """Timeline Extraction: temporal relation classification and consistency research."""
    logging.basicConfig(level=logging.DEBUG if verbose else getattr(logging, log_level))
    ctx.ensure_object(dict)
    ctx.obj["config"] = {}
    ctx.obj["tracking"] = TrackingConfig(backend=tracking, **tracking_options)
    if config:
        with open(config) as file:
            ctx.obj["config"] = yaml.safe_load(file) if Path(config).suffix in {".yaml", ".yml"} else json.load(file)


@cli.command()
@click.option("--data-path", type=click.Path(exists=True), required=True)
@click.option("--output", type=click.Path())
def analyze_data(data_path, output):
    """Analyze a MATRES gold file."""
    data = load_data(Path(data_path))
    analysis = {"document_count": int(data["docid"].nunique()), "relation_count": len(data), "columns": list(data.columns)}
    content = json.dumps(analysis, indent=2)
    if output:
        Path(output).write_text(content + "\n")
    click.echo(content)

@cli.command()
@click.option("--dataset-format", type=click.Choice(["matres", "omnitemp"]), default="matres", show_default=True, help="Source corpus format.")
@click.option("--gold-file", type=click.Path(exists=True, dir_okay=False), help="MATRES gold labels; required for matres.")
@click.option("--docs-dir", type=click.Path(exists=True, file_okay=False), help="TimeML documents; required for matres.")
@click.option("--input-dir", type=click.Path(exists=True, file_okay=False), help="Native OmniTemp JSON split directory; required for omnitemp.")
@click.option("--data-name", help="Dataset key (default: te3-platinum for matres, omnitemp for omnitemp).")
@click.option("--output-dir", type=click.Path(), default="./data/TRC/raw_text")
@click.option("--config-file", type=click.Path(exists=True))
@click.option("--output-config", type=click.Path(), help="Save configuration pointing to the prepared corpus.")
@click.option("--overwrite", is_flag=True)
@click.pass_context
def prepare_data(ctx, dataset_format, gold_file, docs_dir, input_dir, data_name, output_dir, config_file, output_config, overwrite):
    """Prepare pair and multi inputs from MATRES/TimeML or native OmniTemp JSON."""
    if dataset_format == "matres":
        if input_dir is not None:
            raise click.UsageError("--input-dir is only valid for --dataset-format omnitemp.")
        if gold_file is None or docs_dir is None:
            raise click.UsageError("MATRES requires both --gold-file and --docs-dir.")
    else:
        if gold_file is not None or docs_dir is not None:
            raise click.UsageError("OmniTemp uses --input-dir, not --gold-file or --docs-dir.")
        if input_dir is None:
            raise click.UsageError("OmniTemp requires --input-dir.")
    data_name = data_name or ("omnitemp" if dataset_format == "omnitemp" else "te3-platinum")
    try:
        manager = setup_config_manager(config_file, ctx.obj["config"])
        output = Path(output_dir)
        files = {mode: output / f"{mode}_{data_name.lower()}_text_w_relations_prepared.json" for mode in ("pair", "multi")}
        gold_path = output / f"{data_name.lower()}_gold.tsv" if dataset_format == "omnitemp" else Path(gold_file)
        outputs = [*files.values(), gold_path] if dataset_format == "omnitemp" else list(files.values())
        for path in outputs:
            if path.exists() and not overwrite:
                raise click.ClickException(f"Input already exists: {path}; use --overwrite to replace it.")
        if dataset_format == "omnitemp":
            gold, prepared = prepare_omnitemp(Path(input_dir))
        else:
            gold = load_data(gold_path)
            prepared = {mode: generate_training_dataset(gold, Path(docs_dir), mode) for mode in files}
            if any(frame.empty for frame in prepared.values()):
                raise click.ClickException("No annotated pairs matched the TimeML documents.")
        output.mkdir(parents=True, exist_ok=True)
        if dataset_format == "omnitemp":
            gold.to_csv(gold_path, sep="\t", header=False, index=False)
            click.echo(f"gold: {gold_path}")
        for mode, data in prepared.items():
            if dataset_format == "omnitemp":
                files[mode].write_text(json.dumps(data, indent=2))
            else:
                prepare_as_jsonl(data, files[mode], mode)
            click.echo(f"{mode}: {files[mode]}")
        manager.data_config.datasets = dict(manager.data_config.datasets or {})
        manager.data_config.datasets[data_name] = {
            "gold_file": str(gold_path.resolve()),
            "prepared_files": {mode: str(path.resolve()) for mode, path in files.items()},
        }
        manager.research_config.dataset = data_name
        if output_config:
            manager.save_config(output_config)
            click.echo(f"Configuration: {output_config}")
    except (ValueError, OSError) as error:
        raise click.ClickException(str(error)) from error



@cli.group()
def model():
    """Model evaluation and management commands."""


@model.command()
@click.option("--model-name", required=True)
@click.option("--method", type=click.Choice(["zero-shot", "few-shot"]), default=None)
@add_common_evaluation_options
@click.pass_context
def evaluate(ctx, model_name, method, config_file, provider, **options):
    """Evaluate one provider/model with explicit experimental controls."""
    try:
        manager = setup_config_manager(config_file, ctx.obj["config"])
        configured = manager.resolve_model_config(model_name, provider)
        if configured.provider == "huggingface" and options["gpu_device"] is not None:
            configured.device = f"cuda:{options['gpu_device']}"
        data_name = options.pop("data_name")
        experiment = _experiment(manager, model_name, method or manager.evaluation_config.method, data_name=data_name, **options)
        _report(_pipeline(manager, ctx).run_experiment(experiment, configured))
    except (ValueError, OSError) as error:
        raise click.ClickException(str(error)) from error


@model.command()
@click.option("--model-names", multiple=True, required=True)
@click.option("--methods", type=click.Choice(["zero-shot", "few-shot"]), multiple=True)
@add_common_evaluation_options
@click.pass_context
def batch_evaluate(ctx, model_names, methods, config_file, provider, **options):
    """Evaluate the model/method Cartesian product."""
    try:
        manager = setup_config_manager(config_file, ctx.obj["config"])
        methods = methods or (manager.evaluation_config.method,)
        models = {name: manager.resolve_model_config(name, provider) for name in model_names}
        if options["gpu_device"] is not None:
            for configured in models.values():
                if configured.provider == "huggingface":
                    configured.device = f"cuda:{options['gpu_device']}"
        data_name = options.pop("data_name")
        base = _experiment(manager, model_names[0], methods[0], data_name=data_name, **options)
        runner = BatchExperimentRunner(_pipeline(manager, ctx))
        results = runner.run_batch_experiments(list(model_names), list(methods), base, models)
        summary = Path(base.output_dir) / "batch_summary.csv"
        runner.save_batch_summary(results, summary)
        click.echo(f"Summary: {summary}")
        for result in results:
            if result.success:
                _report(result)
        failures = [r.error for r in results if not r.success]
        if failures:
            raise click.ClickException("\n".join(failures))
    except (ValueError, OSError) as error:
        raise click.ClickException(str(error)) from error


@cli.command()
@click.option("--goal", type=click.Choice(GOALS))
@click.option("--provider", "providers", type=click.Choice(PROVIDERS), multiple=True)
@click.option("--method", "methods", type=click.Choice(["zero-shot", "few-shot"]), multiple=True)
@click.option("--model-name", help="Override model ID; select exactly one provider.")
@click.option("--data-name", "dataset", help="Dataset key in configuration.")
@click.option("--output-dir", type=click.Path())
@click.option("--overwrite/--no-overwrite", default=None)
@click.option("--skip-model-eval/--no-skip-model-eval", default=None)
@click.option("--config-file", type=click.Path(exists=True))
@click.option("--plan", "plan_only", is_flag=True, help="Print effective settings without credentials or API calls.")
@click.pass_context
def research(ctx, config_file, plan_only, **selection):
    """Run classification, document consistency, or a controlled pair/multi comparison."""
    try:
        manager = setup_config_manager(config_file, ctx.obj["config"])
        plan = build_research_plan(manager, **selection)
        if plan_only:
            click.echo(json.dumps(describe_research_plan(plan), indent=2))
            return
        pipeline = _pipeline(manager, ctx)
        # Validate all data before the first paid provider request.
        for experiment, configured in plan:
            paths = pipeline._get_file_paths(experiment, configured)
            gold = pipeline._get_labeled_data_path(experiment)
            required = [gold, paths["llm_response"] if experiment.skip_model_eval else paths["raw_text"]]
            for path in required:
                if not path.is_file():
                    raise click.ClickException(f"Missing input: {path}. Configure data.data_path and data.datasets for your corpus.")
        results = [pipeline.run_experiment(experiment, configured) for experiment, configured in plan]
        summary = Path(plan[0][0].output_dir) / "research_summary.csv"
        BatchExperimentRunner(pipeline).save_batch_summary(results, summary)
        click.echo(f"Summary: {summary}")
        for result in results:
            if result.success:
                _report(result)
        failures = [r.error for r in results if not r.success]
        if failures:
            raise click.ClickException("\n".join(failures))
    except (ValueError, OSError, DataError) as error:
        raise click.ClickException(str(error)) from error


@cli.command()
@click.option("--output", type=click.Path(), default="config.yaml")
@click.option("--goal", type=click.Choice(GOALS), default="classification")
@click.option("--provider", type=click.Choice(PROVIDERS), default="openai")
def generate_config(output, goal, provider):
    """Generate a research configuration; credentials are read from the environment."""
    config = create_default_config()
    config["research"]["goal"] = goal
    config["research"]["providers"] = [provider]
    Path(output).write_text(yaml.safe_dump(config, sort_keys=False))
    click.echo(f"Configuration file generated: {output}")


@cli.command()
@click.option("--model-names", multiple=True, default=["gpt-6-luna"])
@click.option("--methods", type=click.Choice(["zero-shot", "few-shot"]), multiple=True, default=["zero-shot", "few-shot"])
@click.option("--mode", type=click.Choice(["pair", "multi", "comb"]), default="comb")
@click.option("--data-name", default="te3-platinum")
@click.option("--output-dir", type=click.Path(), default="./results/demo")
@click.option("--config-file", type=click.Path(exists=True))
@click.pass_context
def demo_pipeline(ctx, model_names, methods, mode, data_name, output_dir, config_file):
    """Run a configured batch against real data (makes paid provider calls)."""
    ctx.invoke(batch_evaluate, model_names=model_names, methods=methods, mode=mode,
               data_name=data_name, output_dir=output_dir, config_file=config_file,
               provider=None, gpu_device=None, use_vague=None, parser_type=None,
               overwrite=None, skip_model_eval=None, full_context=True)


if __name__ == "__main__":
    cli()
