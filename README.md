# Anthropic Prompt Cache test app

This sample runs a sequential multi-turn Messages API conversation and prints the prompt-cache usage fields returned by Anthropic:

- `cache_creation_input_tokens`
- `cache_read_input_tokens`
- `input_tokens`
- `output_tokens`

It uses Claude Haiku by default, enables top-level automatic caching, and adds an explicit cache breakpoint on the stable system context. The generated stable context is intentionally large enough for Haiku prompt caching by default.

## Setup

No package install is required; the app uses only the Python standard library.

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_MESSAGES_API_URL and ANTHROPIC_API_KEY.
python3 prompt_cache_demo.py
```

You can also set variables inline:

```bash
ANTHROPIC_MESSAGES_API_URL="https://api.anthropic.com/v1/messages" \
ANTHROPIC_API_KEY="..." \
python3 prompt_cache_demo.py --turns 20 --interval 60
```

For an Azure API Management endpoint that expects a subscription key header, set:

```bash
ANTHROPIC_API_KEY_HEADER=Ocp-Apim-Subscription-Key
```

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `ANTHROPIC_MESSAGES_API_URL` | `https://api.anthropic.com/v1/messages` | Full Messages API endpoint URL. |
| `ANTHROPIC_API_KEY` | required | API key or APIM subscription key value. |
| `ANTHROPIC_API_KEY_HEADER` | `x-api-key` | Header used for the key. |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5` | Haiku model ID or alias. |
| `ANTHROPIC_VERSION` | `2023-06-01` | Anthropic API version header. |
| `ANTHROPIC_BETA` | unset | Optional beta header. |
| `PROMPT_CACHE_TURNS` | `20` | Number of user/assistant turns. Capped at 20. |
| `PROMPT_CACHE_MAX_TOKENS` | `160` | Max output tokens per turn. |
| `PROMPT_CACHE_TTL` | `5m` | Use `5m` or `1h`. |
| `PROMPT_CACHE_STREAMING` | `false` | Set to `true` to send `"stream": true` and parse SSE events. |
| `PROMPT_CACHE_CONTEXT_REPETITIONS` | `80` | Size of stable generated context. Lower values may fall below Haiku's cacheable token minimum. |
| `PROMPT_CACHE_INTERVAL_SECONDS` | `0` | Seconds to wait between turns. `--interval 60` waits 60 seconds before the next turn. |
| `PROMPT_CACHE_OUTPUT_JSONL` | unset | Optional JSONL output path. |

## Dry run

Use `--dry-run` to confirm the payload shape without calling the API:

```bash
python3 prompt_cache_demo.py --dry-run --turns 2
```

## Streaming mode

Regular, non-streaming responses are used by default. Enable Messages API streaming with:

```bash
python3 prompt_cache_demo.py --streaming true
```

Disable it explicitly with:

```bash
python3 prompt_cache_demo.py --streaming false
```

The same setting can be controlled with `PROMPT_CACHE_STREAMING=true` in `.env`.

## Interpreting results

The first turn is usually a cache write. Later turns should show `HIT` with non-zero `cache_read_input_tokens` if the endpoint supports prompt caching and the prompt is large enough for the selected Haiku model. If all cache fields stay at zero, increase `PROMPT_CACHE_CONTEXT_REPETITIONS` or confirm that your gateway forwards prompt-cache fields and headers unchanged.
