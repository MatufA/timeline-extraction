"""Normalize the single-choice chat responses used by hosted providers."""


def chat_messages(prompt):
    return [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt


def chat_response(response, trail):
    choices = response.choices
    if len(choices) != 1:
        raise ValueError("Expected one completion per trial")
    choice = choices[0]
    message = choice.message
    refusal = getattr(message, "refusal", None)
    if refusal:
        raise ValueError(f"Provider refused the request: {refusal}")
    if getattr(choice, "finish_reason", None) in {"length", "content_filter"}:
        raise ValueError(f"Provider did not complete the response: {choice.finish_reason}")
    content = message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Provider returned no text content")
    usage = getattr(response, "usage", None)
    return {
        "trail": trail,
        "response": content,
        "content": content,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
