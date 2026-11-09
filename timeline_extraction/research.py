"""Research-goal selection shared by CLI and programmatic experiments."""

from dataclasses import asdict
from typing import Optional, Sequence

from timeline_extraction.config import ConfigManager, ModelConfig, ResearchConfig
from timeline_extraction.pipeline import PipelineExperimentConfig


def build_research_plan(
    manager: ConfigManager,
    *,
    goal: Optional[str] = None,
    providers: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    dataset: Optional[str] = None,
    model_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    overwrite: Optional[bool] = None,
    skip_model_eval: Optional[bool] = None,
) -> list[tuple[PipelineExperimentConfig, ModelConfig]]:
    """Select pairwise accuracy, document consistency, or their comparison.

    Provider/model IDs remain configuration, not a hard-coded model allowlist.
    A custom model ID requires exactly one provider to avoid ambiguous routing.
    """
    defaults = manager.research_config
    selection = ResearchConfig(
        goal=goal or defaults.goal,
        providers=list(providers or defaults.providers),
        methods=list(methods or defaults.methods),
        dataset=dataset or defaults.dataset,
    )
    if model_name and len(selection.providers) != 1:
        raise ValueError("--model-name requires exactly one --provider")
    modes = {
        "classification": ["pair"],
        "consistency": ["multi"],
        "comparison": ["pair", "multi"],
    }[selection.goal]
    evaluation = manager.evaluation_config
    plan = []
    for provider in dict.fromkeys(selection.providers):
        model = manager.resolve_model_config(model_name, provider)
        for method in dict.fromkeys(selection.methods):
            for mode in modes:
                experiment = PipelineExperimentConfig(
                    model_name=model.name,
                    method=method,
                    mode=mode,
                    data_name=selection.dataset,
                    use_vague=evaluation.use_vague,
                    parser_type="json" if mode == "multi" else "label",
                    overwrite=evaluation.overwrite if overwrite is None else overwrite,
                    skip_model_eval=evaluation.skip_model_eval if skip_model_eval is None else skip_model_eval,
                    output_dir=output_dir or manager.data_config.results_path,
                    research_goal=selection.goal,
                )
                plan.append((experiment, model))
    return plan


def describe_research_plan(plan: list[tuple[PipelineExperimentConfig, ModelConfig]]) -> list[dict]:
    """Expose reproducible settings without leaking configured credentials."""
    return [
        {
            "experiment": asdict(experiment),
            "model": {key: value for key, value in asdict(model).items() if key != "api_key"},
        }
        for experiment, model in plan
    ]
