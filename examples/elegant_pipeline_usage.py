"""Run research-goal experiments through the programmatic API.

Usage:
    uv run python -m examples.elegant_pipeline_usage --config config.yaml --plan
    uv run python -m examples.elegant_pipeline_usage --config config.yaml --goal comparison

Supply corpus paths and provider credentials before running without --plan.
"""

import argparse
import json
from pathlib import Path

from timeline_extraction.config import ConfigManager
from timeline_extraction.pipeline import BatchExperimentRunner, TimelineExtractionPipeline
from timeline_extraction.research import build_research_plan, describe_research_plan


def main():
    parser = argparse.ArgumentParser(description="Run temporal-relation research by goal.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--goal", choices=["classification", "consistency", "comparison"])
    parser.add_argument("--provider", action="append", choices=["openai", "google", "together", "groq", "huggingface", "antigravity", "codex", "claude"])
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    manager = ConfigManager()
    manager.load_config(args.config)
    plan = build_research_plan(manager, goal=args.goal, providers=args.provider)
    if args.plan:
        print(json.dumps(describe_research_plan(plan), indent=2))
        return
    pipeline = TimelineExtractionPipeline(manager)
    results = []
    for experiment, model in plan:
        result = pipeline.run_experiment(experiment, model)
        results.append(result)
        print(f"{result.provider}/{result.mode}/{result.method}: {result.metrics_path or result.error}")
    summary = Path(manager.data_config.results_path) / "research_summary.csv"
    BatchExperimentRunner(pipeline).save_batch_summary(results, summary)
    failures = [result.error for result in results if not result.success]
    if failures:
        raise RuntimeError("\n".join(failures))
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
