"""Opt-in experiment and generation tracing; SDKs are imported only when enabled."""

import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrackingConfig:
    backend: str = "none"
    langsmith_api_key: str | None = field(default=None, repr=False)
    langsmith_project: str = "timeline-extraction"
    langsmith_endpoint: str | None = None
    langfuse_public_key: str | None = field(default=None, repr=False)
    langfuse_secret_key: str | None = field(default=None, repr=False)
    langfuse_base_url: str | None = None

    def __post_init__(self):
        if self.backend not in {"none", "langsmith", "langfuse", "both"}:
            raise ValueError("Tracking must be none, langsmith, langfuse, or both")

    @classmethod
    def from_env(cls):
        return cls(
            backend=os.getenv("TIMELINE_TRACKING", "none"),
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY"),
            langsmith_project=os.getenv("LANGSMITH_PROJECT", "timeline-extraction"),
            langsmith_endpoint=os.getenv("LANGSMITH_ENDPOINT"),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            langfuse_base_url=os.getenv("LANGFUSE_BASE_URL"),
        )


class Observation:
    def __init__(self, langsmith_run=None, langfuse_span=None):
        self.langsmith_run = langsmith_run
        self.langfuse_span = langfuse_span

    def finish(self, output, *, error=None, usage=None):
        if self.langsmith_run is not None:
            smith_output = output
            if usage:
                smith_output = {
                    **output,
                    "usage_metadata": {f"{key}_tokens": value for key, value in usage.items()},
                }
            self.langsmith_run.end(outputs=smith_output, error=error)
        if self.langfuse_span is not None:
            fields = {"output": output}
            if usage:
                fields["usage_details"] = usage
            if error:
                fields.update(level="ERROR", status_message=error)
            self.langfuse_span.update(**fields)


class ExperimentTracker:
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.langsmith = None
        self.langfuse = None
        if config.backend in {"langsmith", "both"} and not config.langsmith_api_key:
            raise ValueError("LangSmith tracking requires LANGSMITH_API_KEY or --langsmith-api-key")
        if config.backend in {"langfuse", "both"} and not (
            config.langfuse_public_key and config.langfuse_secret_key
        ):
            raise ValueError("Langfuse tracking requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY (or their CLI flags)")
        try:
            if config.backend in {"langsmith", "both"}:
                from langsmith import Client

                self.langsmith = Client(
                    api_key=config.langsmith_api_key, api_url=config.langsmith_endpoint,
                )
            if config.backend in {"langfuse", "both"}:
                from langfuse import Langfuse

                self.langfuse = Langfuse(
                    public_key=config.langfuse_public_key,
                    secret_key=config.langfuse_secret_key,
                    base_url=config.langfuse_base_url,
                )
        except ImportError as error:
            raise ValueError("Tracking SDKs are not installed; run uv sync --extra tracking") from error

    @contextmanager
    def span(self, name, inputs, metadata, *, model=None):
        with ExitStack() as stack:
            smith_run = None
            fuse_span = None
            if self.langsmith is not None:
                from langsmith import trace, tracing_context

                stack.enter_context(tracing_context(enabled=True))
                smith_run = stack.enter_context(trace(
                    name, run_type="llm" if model else "chain",
                    inputs=inputs, metadata=metadata, client=self.langsmith,
                    project_name=self.config.langsmith_project,
                ))
            if self.langfuse is not None:
                fuse_span = stack.enter_context(self.langfuse.start_as_current_observation(
                    name=name, as_type="generation" if model else "span",
                    input=inputs, metadata=metadata, **({"model": model} if model else {}),
                ))
            yield Observation(smith_run, fuse_span)

    def generate(self, model, prompt, metadata):
        """Trace only a real request, not checkpoint/cache replay or parsed relation rows."""
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        with self.span("model.generate", {"messages": messages}, metadata, model=metadata["ls_model_name"]) as span:
            responses = model.prepare_response(model.generate_response(prompt))
            usage = {}
            for source, target in (
                ("prompt_tokens", "input"), ("completion_tokens", "output"), ("total_tokens", "total"),
            ):
                counts = [row.get(source) for row in responses]
                if counts and all(isinstance(value, int) for value in counts):
                    usage[target] = sum(counts)
            span.finish({
                "choices": [
                    {"index": row["trail"], "message": {"role": "assistant", "content": row.get("content", row["response"])}}
                    for row in responses
                ],
                "responses": responses,
            }, usage=usage)
            return responses

    def flush(self):
        if self.langsmith is not None:
            self.langsmith.flush()
        if self.langfuse is not None:
            self.langfuse.flush()
