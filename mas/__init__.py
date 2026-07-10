import re
import string

import openai

from mas.llm import OpenRouterChatModel
from mas.generation import load_scenarios
from mas.builder import build_config
from mas.runner import run_scenario
from concordia.contrib.language_models.openai import base_gpt_model
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

class OpenRouterLanguageModel(base_gpt_model.BaseGPTModel):
    """GPT-style model routed through OpenRouter.
    Overrides sample_choice with tolerant answer parsing: the base
    implementation requires an exact string match against the option
    letter, which fails intermittently when the model adds trailing
    punctuation or surrounding text (e.g. "a." or "The answer is (a).").
    """
    def __init__(
        self,
        model_name: str,
        *,
        api_key: str,
        measurements: measurements_lib.Measurements | None = None,
        channel: str = language_model.DEFAULT_STATS_CHANNEL
    ):
        client = openai.OpenAI(api_key=api_key, base_url=OPENROUTER_BASE)
        super().__init__(
            model_name=model_name,
            client=client,
            measurements=measurements,
            channel=channel,
        )
    def sample_choice(self, prompt, responses, *, seed=None):
        full_prompt = (
            prompt
            + '\nRespond EXACTLY with one of the following strings:\n'
            + '\n'.join(responses)
            + '.'
        )
        last_sample = ''
        for attempts in range(base_gpt_model._MAX_MULTIPLE_CHOICE_ATTEMPTS):
          last_sample = self._sample_text(
              full_prompt,
              reasoning_effort='medium',
              verbosity=self._verbosity,
              temperature=1.0,
              seed=seed,
          )
          cleaned = last_sample.strip().strip(string.punctuation + ' ').lower()
          matches = [r for r in responses if r.lower() == cleaned]
          if len(matches) != 1:
            # Fall back to a standalone-token search, e.g. pull "a" out of
            # "The answer is (a)." instead of requiring a bare "a".
            found = {
                r for r in responses
                if re.search(
                    rf'(?<![a-zA-Z0-9]){re.escape(r)}(?![a-zA-Z0-9])',
                    last_sample,
                    re.IGNORECASE,
                )
            }
            if len(found) == 1:
              matches = list(found)
          if len(matches) == 1:
            idx = responses.index(matches[0])
            if self._measurements is not None:
              self._measurements.publish_datum(
                  self._channel, {'choices_calls': attempts}
              )
            return idx, responses[idx], {}
        raise language_model.InvalidResponseError((
            f'Too many multiple choice attempts.\nLast attempt: '
            f'{last_sample}, extracted: {last_sample.strip()}'
        ))