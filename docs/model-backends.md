# Model backends

inkmoth talks to any server that implements the OpenAI Chat Completions API with tool calling:
vLLM, SGLang, llama.cpp `llama-server`, Ollama, LM Studio, TGI, or a hosted API.

```bash
INKMOTH_LLM_BASE_URL=http://localhost:8000/v1   # must end at the /v1 prefix
INKMOTH_LLM_MODEL=your-model-name               # required; `inkmoth doctor` lists what the server offers
INKMOTH_LLM_API_KEY=EMPTY                       # local servers usually ignore it
```

Run `inkmoth doctor` first. It checks that the model is listed, that it actually emits a tool call, and
that the search backend answers. A model that cannot call tools reliably cannot run the researcher
phase; that is the main requirement, not size.

## Thinking / reasoning toggles

The pipeline has three toggles: `INKMOTH_THINKING_PLANNER` (default on), `INKMOTH_THINKING_RESEARCHER` and
`INKMOTH_THINKING_SYNTHESIZER` (default off), plus `--no-thinking` on the CLI. Servers disagree on how
reasoning is switched, so inkmoth sends nothing for it unless you say how:

| Setting | Merged into the request when |
|---|---|
| `INKMOTH_LLM_EXTRA_BODY_THINKING` | the phase has thinking on |
| `INKMOTH_LLM_EXTRA_BODY_NO_THINKING` | the phase has thinking off |

Both take a JSON object and default to `{}`. With both empty the toggles do nothing and every request
is a plain OpenAI request (the real OpenAI API rejects unknown body fields with HTTP 400).

| Server / model | `..._THINKING` | `..._NO_THINKING` |
|---|---|---|
| vLLM or SGLang with Qwen3 / GLM-style chat templates | `{"chat_template_kwargs": {"enable_thinking": true}}` | `{"chat_template_kwargs": {"enable_thinking": false}}` |
| OpenAI-style reasoning models | `{"reasoning_effort": "high"}` | `{"reasoning_effort": "low"}` |
| Anything else / unsure | leave empty | leave empty |

Set the "no thinking" body too when the model thinks by default (Qwen3 does): otherwise the
researcher and synthesizer phases think anyway and spend far more tokens.

If a phase hits the output token limit while thinking, it retries once with thinking off.

## Models tested

The evals (`evals/RESULTS-*.md`) were run with GLM-5.3-Flash (NVFP4) on vLLM, using the
`chat_template_kwargs` bodies above with thinking off for every role. Other tool-calling models should work;
`inkmoth doctor` is the quickest check.
