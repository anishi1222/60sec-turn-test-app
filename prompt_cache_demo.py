#!/usr/bin/env python3
"""Exercise Anthropic Messages API prompt caching over a multi-turn chat."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
MAX_TURNS = 20

REFERENCE_PARAGRAPH = (
    "Reference section {index}: This is stable prompt-cache test material. "
    "It describes a fictional API Management gateway, request governance rules, "
    "observability requirements, retry expectations, quota handling, and "
    "response validation criteria. The wording is intentionally unchanged on "
    "every request so the model can reuse a cached prefix while the conversation "
    "history grows. Treat these instructions as background context and answer "
    "the user's short turn-specific question using concise practical language."
)


@dataclass(frozen=True)
class Config:
    url: str
    api_key: str | None
    api_key_header: str
    anthropic_version: str
    anthropic_beta: str | None
    model: str
    turns: int
    max_tokens: int
    ttl: str
    timeout_seconds: float
    interval_seconds: float
    context_repetitions: int
    streaming: bool
    output_jsonl: Path | None
    dry_run: bool


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc


def parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Boolean value must be true or false, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run up to 20 sequential Anthropic Messages API turns and print "
            "prompt-cache usage counters."
        )
    )
    parser.add_argument("--env-file", default=".env", help="Optional .env file to load first.")
    parser.add_argument("--no-dotenv", action="store_true", help="Do not load an .env file.")
    parser.add_argument("--url", default=None, help="Messages API endpoint URL.")
    parser.add_argument("--api-key", default=None, help="API key. Prefer ANTHROPIC_API_KEY.")
    parser.add_argument("--api-key-header", default=None, help="API key header name.")
    parser.add_argument("--model", default=None, help="Model ID. Defaults to Claude Haiku.")
    parser.add_argument("--turns", type=int, default=None, help="Number of turns, 1-20.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max output tokens per turn.")
    parser.add_argument(
        "--ttl",
        default=None,
        choices=("5m", "1h"),
        help="Prompt cache TTL. 5m uses Anthropic's default; 1h requests the longer TTL.",
    )
    parser.add_argument(
        "--context-repetitions",
        type=int,
        default=None,
        help="Number of repeated stable context paragraphs.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=None, help="HTTP timeout.")
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Seconds to wait between turns.",
    )
    parser.add_argument(
        "--streaming",
        default=None,
        help="Set to true to use Messages API streaming mode, or false for regular responses.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Optional path for per-turn response and usage JSON Lines.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build payloads without calling the API.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    if not args.no_dotenv:
        load_env_file(Path(args.env_file))

    turns = args.turns if args.turns is not None else env_int("PROMPT_CACHE_TURNS", MAX_TURNS)
    if turns < 1 or turns > MAX_TURNS:
        raise ValueError(f"turns must be between 1 and {MAX_TURNS}; got {turns}")

    max_tokens = (
        args.max_tokens if args.max_tokens is not None else env_int("PROMPT_CACHE_MAX_TOKENS", 160)
    )
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be positive; got {max_tokens}")

    context_repetitions = (
        args.context_repetitions
        if args.context_repetitions is not None
        else env_int("PROMPT_CACHE_CONTEXT_REPETITIONS", 80)
    )
    if context_repetitions < 1:
        raise ValueError(f"context_repetitions must be positive; got {context_repetitions}")

    interval_seconds = (
        args.interval
        if args.interval is not None
        else env_float("PROMPT_CACHE_INTERVAL_SECONDS", 0.0)
    )
    if interval_seconds < 0:
        raise ValueError(f"interval must be zero or positive; got {interval_seconds}")

    output_jsonl_value = args.output_jsonl or os.environ.get("PROMPT_CACHE_OUTPUT_JSONL")

    return Config(
        url=args.url or os.environ.get("ANTHROPIC_MESSAGES_API_URL", DEFAULT_MESSAGES_URL),
        api_key=args.api_key or os.environ.get("ANTHROPIC_API_KEY"),
        api_key_header=args.api_key_header
        or os.environ.get("ANTHROPIC_API_KEY_HEADER", "x-api-key"),
        anthropic_version=os.environ.get("ANTHROPIC_VERSION", DEFAULT_ANTHROPIC_VERSION),
        anthropic_beta=os.environ.get("ANTHROPIC_BETA") or None,
        model=args.model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL),
        turns=turns,
        max_tokens=max_tokens,
        ttl=args.ttl or os.environ.get("PROMPT_CACHE_TTL", "5m"),
        timeout_seconds=(
            args.timeout_seconds
            if args.timeout_seconds is not None
            else env_float("PROMPT_CACHE_TIMEOUT_SECONDS", 120.0)
        ),
        interval_seconds=interval_seconds,
        context_repetitions=context_repetitions,
        streaming=parse_bool(args.streaming, parse_bool(os.environ.get("PROMPT_CACHE_STREAMING"))),
        output_jsonl=Path(output_jsonl_value) if output_jsonl_value else None,
        dry_run=args.dry_run,
    )


def cache_control(ttl: str) -> dict[str, str]:
    value = {"type": "ephemeral"}
    if ttl == "1h":
        value["ttl"] = "1h"
    elif ttl != "5m":
        raise ValueError("PROMPT_CACHE_TTL must be either 5m or 1h")
    return value


def build_reference_context(repetitions: int) -> str:
    return "\n".join(REFERENCE_PARAGRAPH.format(index=index) for index in range(1, repetitions + 1))


def build_user_text(turn: int, total_turns: int) -> str:
    return (
        f"Turn {turn}/{total_turns}: summarize two cache-relevant facts from the stable "
        "reference context, then ask one brief follow-up question for the next turn."
    )


def text_block(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def build_payload(config: Config, messages: list[dict[str, Any]], reference_context: str) -> dict[str, Any]:
    cc = cache_control(config.ttl)
    payload = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "cache_control": cc,
        "system": [
            {
                "type": "text",
                "text": (
                    "You are a concise assistant for an Anthropic prompt-cache behavior test. "
                    "Keep answers short so the test focuses on input-side cache counters.\n\n"
                    + reference_context
                ),
                "cache_control": cc,
            }
        ],
        "messages": messages,
    }
    if config.streaming:
        payload["stream"] = True
    return payload


def request_headers(config: Config) -> dict[str, str]:
    if not config.api_key:
        raise ValueError("ANTHROPIC_API_KEY is required unless --dry-run is used")

    headers = {
        "accept": "text/event-stream" if config.streaming else "application/json",
        "content-type": "application/json",
        config.api_key_header: config.api_key,
        "anthropic-version": config.anthropic_version,
    }
    if config.anthropic_beta:
        headers["anthropic-beta"] = config.anthropic_beta
    return headers


def send_message(config: Config, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        config.url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers(config),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {config.url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to call {config.url}: {exc.reason}") from exc


def merge_usage(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, int):
            target[key] = value


def accumulate_stream_event(
    event: dict[str, Any],
    message: dict[str, Any],
    content_by_index: dict[int, dict[str, Any]],
    usage: dict[str, Any],
) -> None:
    event_type = event.get("type")
    if event_type == "message_start":
        started_message = event.get("message", {})
        if isinstance(started_message, dict):
            message.update({key: value for key, value in started_message.items() if key != "content"})
            started_usage = started_message.get("usage", {})
            if isinstance(started_usage, dict):
                merge_usage(usage, started_usage)
        return

    if event_type == "content_block_start":
        index = event.get("index")
        block = event.get("content_block", {})
        if isinstance(index, int) and isinstance(block, dict):
            content_by_index[index] = dict(block)
        return

    if event_type == "content_block_delta":
        index = event.get("index")
        delta = event.get("delta", {})
        if not isinstance(index, int) or not isinstance(delta, dict):
            return

        block = content_by_index.setdefault(index, {"type": "text", "text": ""})
        delta_type = delta.get("type")
        if delta_type == "text_delta":
            block["text"] = block.get("text", "") + delta.get("text", "")
        elif delta_type == "thinking_delta":
            block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
        elif delta_type == "signature_delta":
            block["signature"] = delta.get("signature", "")
        elif delta_type == "input_json_delta":
            block["_partial_json"] = block.get("_partial_json", "") + delta.get("partial_json", "")
        return

    if event_type == "message_delta":
        delta = event.get("delta", {})
        if isinstance(delta, dict):
            message.update(delta)
        delta_usage = event.get("usage", {})
        if isinstance(delta_usage, dict):
            merge_usage(usage, delta_usage)
        return

    if event_type == "error":
        error = event.get("error", {})
        if isinstance(error, dict):
            raise RuntimeError(f"Streaming error: {error.get('type')}: {error.get('message')}")
        raise RuntimeError(f"Streaming error: {event}")


def parse_sse_stream(response: Any) -> dict[str, Any]:
    message: dict[str, Any] = {"content": [], "usage": {}}
    content_by_index: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] = {}
    event_name: str | None = None
    data_lines: list[str] = []

    def dispatch() -> None:
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = None
            return
        raw_data = "\n".join(data_lines).strip()
        dispatched_event_name = event_name
        event_name = None
        data_lines = []
        if not raw_data or raw_data == "[DONE]":
            return
        try:
            parsed = json.loads(raw_data)
        except json.JSONDecodeError as exc:
            event_label = dispatched_event_name or "message"
            raise RuntimeError(f"Invalid SSE {event_label} data: {raw_data[:200]!r}") from exc
        if isinstance(parsed, dict):
            accumulate_stream_event(parsed, message, content_by_index, usage)

    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if line == "":
            dispatch()
        elif line.startswith("event:"):
            event_name = line.partition(":")[2].strip()
        elif line.startswith("data:"):
            data_lines.append(line.partition(":")[2].lstrip())

    dispatch()

    content = [content_by_index[index] for index in sorted(content_by_index)]
    for block in content:
        partial_json = block.pop("_partial_json", None)
        if isinstance(partial_json, str) and partial_json:
            try:
                block["input"] = json.loads(partial_json)
            except json.JSONDecodeError:
                block["input"] = partial_json

    message["content"] = content
    message["usage"] = usage
    return message


def send_message_streaming(config: Config, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        config.url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers(config),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            return parse_sse_stream(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {config.url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to call {config.url}: {exc.reason}") from exc


def assistant_blocks(response: dict[str, Any]) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    for block in response.get("content", []):
        if block.get("type") == "text" and block.get("text", "").strip():
            blocks.append(text_block(block["text"]))
    if not blocks:
        blocks.append(text_block("(No text content returned.)"))
    return blocks


def usage_value(usage: dict[str, Any], key: str) -> int:
    value = usage.get(key, 0)
    return value if isinstance(value, int) else 0


def print_row(turn: int, elapsed_ms: int, usage: dict[str, Any]) -> None:
    input_tokens = usage_value(usage, "input_tokens")
    cache_creation = usage_value(usage, "cache_creation_input_tokens")
    cache_read = usage_value(usage, "cache_read_input_tokens")
    output_tokens = usage_value(usage, "output_tokens")
    event = "HIT" if cache_read else "WRITE" if cache_creation else "NONE"
    print(
        f"{turn:>2} | {event:<5} | "
        f"input={input_tokens:>5} "
        f"cache_write={cache_creation:>5} "
        f"cache_read={cache_read:>5} "
        f"output={output_tokens:>4} "
        f"elapsed_ms={elapsed_ms:>6}"
    )


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def dry_run(config: Config, reference_context: str) -> None:
    messages = [{"role": "user", "content": [text_block(build_user_text(1, config.turns))]}]
    payload = build_payload(config, messages, reference_context)
    encoded = json.dumps(payload, ensure_ascii=False)
    print("dry_run=true")
    print(f"url={config.url}")
    print(f"model={config.model}")
    print(f"turns={config.turns}")
    print(f"streaming={str(config.streaming).lower()}")
    print(f"interval_seconds={config.interval_seconds:g}")
    print(f"context_repetitions={config.context_repetitions}")
    print(f"payload_bytes={len(encoded.encode('utf-8'))}")
    print(f"top_level_cache_control={payload['cache_control']}")
    print(f"system_blocks={len(payload['system'])}")
    print(f"message_count={len(payload['messages'])}")


def run(config: Config) -> None:
    reference_context = build_reference_context(config.context_repetitions)
    if config.dry_run:
        dry_run(config, reference_context)
        return

    if config.output_jsonl and config.output_jsonl.exists():
        config.output_jsonl.unlink()

    messages: list[dict[str, Any]] = []
    total_cache_creation = 0
    total_cache_read = 0
    total_input = 0
    total_output = 0
    hit_turns = 0

    print("turn | event | usage")
    print("-----+-------+---------------------------------------------------------------")
    for turn in range(1, config.turns + 1):
        messages.append({"role": "user", "content": [text_block(build_user_text(turn, config.turns))]})
        payload = build_payload(config, messages, reference_context)

        started_at = time.monotonic()
        response = send_message_streaming(config, payload) if config.streaming else send_message(config, payload)
        elapsed_ms = int((time.monotonic() - started_at) * 1000)

        usage = response.get("usage", {})
        cache_creation = usage_value(usage, "cache_creation_input_tokens")
        cache_read = usage_value(usage, "cache_read_input_tokens")
        input_tokens = usage_value(usage, "input_tokens")
        output_tokens = usage_value(usage, "output_tokens")

        total_cache_creation += cache_creation
        total_cache_read += cache_read
        total_input += input_tokens
        total_output += output_tokens
        hit_turns += 1 if cache_read else 0

        print_row(turn, elapsed_ms, usage)

        if config.output_jsonl:
            write_jsonl(
                config.output_jsonl,
                {
                    "turn": turn,
                    "elapsed_ms": elapsed_ms,
                    "usage": usage,
                    "response_id": response.get("id"),
                    "stop_reason": response.get("stop_reason"),
                    "content": response.get("content"),
                },
            )

        messages.append({"role": "assistant", "content": assistant_blocks(response)})

        if config.interval_seconds and turn < config.turns:
            print(f"waiting_seconds={config.interval_seconds:g}")
            time.sleep(config.interval_seconds)

    print("-----+-------+---------------------------------------------------------------")
    print(
        "summary: "
        f"turns={config.turns} "
        f"hit_turns={hit_turns} "
        f"cache_write_tokens={total_cache_creation} "
        f"cache_read_tokens={total_cache_read} "
        f"uncached_input_tokens={total_input} "
        f"output_tokens={total_output}"
    )
    if config.output_jsonl:
        print(f"jsonl={config.output_jsonl}")


def main() -> int:
    try:
        args = parse_args()
        config = build_config(args)
        run(config)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
