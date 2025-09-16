import os
from together import Together

from timeline_extraction.models.LLModel import LLModel

# client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
#
# stream = client.chat.completions.create(
#     model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
#     messages=[{"role": "user", "content": "What are some fun things to do in New York?"}],
#     stream=True,
# )
#
# for chunk in stream:
#     print(chunk.choices[0].delta.content or "", end="", flush=True)


class TogetherAIClient(LLModel):
    def __init__(self, model_name: str, n_trails: int = 5):
        model = self._initial_client()
        self.model_name: str = model_name
        super().__init__(model, each_trail=False, each_doc=True, n_trails=n_trails)

    @staticmethod
    def _initial_client() -> Together:
        return Together(api_key=os.environ.get("TOGETHER_API_KEY"))

    def generate_response(self, prompt):
        return self.model.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_name,
        )

    def prepare_response(self, response):
        return {
            "response": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }


if __name__ == "__main__":
    model = TogetherAIClient(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
