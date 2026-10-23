"""Step 3: Preflight the vLLM endpoint without touching the server.

Writes `runs/preflight.md` with:
- /health
- /v1/models
- /version (vLLM version)
- one thinking-on request: confirms reasoning is in `message.reasoning`,
  content is valid JSON, reasoning_tokens is populated (or not), and
  thinking_token_budget is accepted (or not)
- one thinking-off request: confirms content is valid JSON and reasoning
  is empty

Exits non-zero if any check indicates the server needs reconfiguration.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

from . import config
from .client import extract_reasoning, reasoning_tokens, sync_client
from .sampling import chat_kwargs, default_timeout
from .schema import ProverbTranslation, response_format

TRANSLATION_MODEL = config.TRANSLATION_MODEL

EXAMPLE_PROVERB = {
    "ko": "굼벵이 천장하듯",
    "definition_ko": (
        "굼벵이는 느리므로 무덤을 옮기자면 오래 걸린다는 뜻으로, "
        "어리석은 사람이 일을 지체하며 좀처럼 성사시키지 못함을 비유적으로 이르는 말."
    ),
}


def _get_json(base_url: str, path: str) -> tuple[int, object]:
    url = base_url.rstrip("/v1") + path
    with httpx.Client(timeout=10) as hc:
        r = hc.get(url)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, r.text


def check_health(base_url: str) -> tuple[bool, str]:
    try:
        with httpx.Client(timeout=10) as hc:
            r = hc.get(base_url.rstrip("/v1") + "/health")
        return r.status_code == 200, f"GET /health → {r.status_code}"
    except Exception as e:
        return False, f"GET /health failed: {e!r}"


def check_models(base_url: str) -> tuple[bool, str]:
    try:
        with httpx.Client(timeout=10) as hc:
            r = hc.get(base_url + "/models")
        data = r.json().get("data", [])
        ids = [m.get("id") for m in data]
        ok = TRANSLATION_MODEL in ids
        return ok, f"GET /v1/models → {ids}"
    except Exception as e:
        return False, f"GET /v1/models failed: {e!r}"


def check_version(base_url: str) -> tuple[str | None, str]:
    try:
        with httpx.Client(timeout=10) as hc:
            r = hc.get(base_url.rstrip("/v1") + "/version")
        data = r.json() if r.status_code == 200 else {}
        v = data.get("version") if isinstance(data, dict) else None
        return v, f"GET /version → {r.status_code}: {data}"
    except Exception as e:
        return None, f"GET /version failed: {e!r}"


def _make_request(client, thinking: bool, user_msg: str, system_prompt: str) -> tuple[object, dict]:
    kw = chat_kwargs(thinking=thinking)
    rf = response_format(ProverbTranslation, "ProverbTranslation")
    req_params = {
        "model": TRANSLATION_MODEL,
        "temperature": kw["temperature"],
        "top_p": kw["top_p"],
        "presence_penalty": kw["presence_penalty"],
        "max_tokens": kw["max_tokens"],
        "seed": 42,
        "extra_body": kw["extra_body"],
    }
    resp = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        response_format=rf,
        timeout=default_timeout(thinking),
        **req_params,
    )
    return resp, req_params


def run_preflight() -> int:
    config.ensure_dirs()
    base = config.qwen_base_url()
    lines: list[str] = []
    lines.append(f"# Preflight — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- `QWEN_BASE_URL`: `{base}`")

    health_ok, health_msg = check_health(base)
    lines.append(f"- {health_msg}")
    if not health_ok:
        lines.append("")
        lines.append("**/health is not OK. Stopping.**")
        Path(config.RUNS / "preflight.md").write_text("\n".join(lines), encoding="utf-8")
        return 1

    models_ok, models_msg = check_models(base)
    lines.append(f"- {models_msg}")
    if not models_ok:
        lines.append("")
        lines.append(f"**Model `{TRANSLATION_MODEL}` not listed. Stopping.**")
        Path(config.RUNS / "preflight.md").write_text("\n".join(lines), encoding="utf-8")
        return 1

    vllm_version, version_msg = check_version(base)
    lines.append(f"- {version_msg}")

    # Thinking-on request.
    from .prompts import load as load_prompt, user_message_translate
    sys_prompt = load_prompt("p1-zero")
    user_msg = user_message_translate(EXAMPLE_PROVERB["ko"], EXAMPLE_PROVERB["definition_ko"], nk=False)

    client = sync_client(base)

    budget_accepted = True
    try:
        resp_on, req_on = _make_request(client, thinking=True, user_msg=user_msg, system_prompt=sys_prompt)
    except Exception as e:
        msg = str(e)
        if "thinking_token_budget" in msg:
            budget_accepted = False
            lines.append("- thinking-on initial request: server rejected `thinking_token_budget`. Retrying without it.")
            # Retry without the budget by dropping it from extra_body.
            kw = chat_kwargs(thinking=True)
            kw["extra_body"].pop("thinking_token_budget", None)
            rf = response_format(ProverbTranslation, "ProverbTranslation")
            resp_on = client.chat.completions.create(
                model=TRANSLATION_MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ],
                response_format=rf,
                timeout=default_timeout(True),
                temperature=kw["temperature"],
                top_p=kw["top_p"],
                presence_penalty=kw["presence_penalty"],
                max_tokens=kw["max_tokens"],
                seed=42,
                extra_body=kw["extra_body"],
            )
        else:
            lines.append(f"- **thinking-on request failed**: {msg}")
            Path(config.RUNS / "preflight.md").write_text("\n".join(lines), encoding="utf-8")
            return 1

    choice = resp_on.choices[0]
    msg_on = choice.message
    reasoning_on = extract_reasoning(msg_on)
    content_on = msg_on.content or ""
    rtok_on = reasoning_tokens(resp_on.usage)
    reasoning_tok_present = rtok_on is not None

    content_valid_json = False
    try:
        ProverbTranslation.model_validate_json(content_on.strip())
        content_valid_json = True
    except Exception as e:
        parse_err = str(e)
    else:
        parse_err = ""

    has_think_tag = "<think>" in content_on

    lines.append("")
    lines.append("## Thinking ON")
    lines.append(f"- `message.reasoning` non-empty: {bool(reasoning_on)} (len={len(reasoning_on)})")
    lines.append(f"- `message.content` validates against ProverbTranslation: {content_valid_json}")
    if not content_valid_json:
        lines.append(f"  - parse error: `{parse_err}`")
    lines.append(f"- `usage.completion_tokens_details.reasoning_tokens` present: {reasoning_tok_present}"
                 + (f" ({rtok_on})" if reasoning_tok_present else ""))
    lines.append(f"- `thinking_token_budget` accepted: {budget_accepted}")
    lines.append(f"- content starts with `<think>`: {has_think_tag}")

    # Fail hard on the two failure modes the plan lists.
    hard_fail = False
    if (not reasoning_on) and content_valid_json and not has_think_tag:
        lines.append("")
        lines.append("**Reasoning is empty while content is bare JSON. "
                     "The schema is suppressing thinking — server needs `--reasoning-parser qwen3`. Stop.**")
        hard_fail = True
    if has_think_tag:
        lines.append("")
        lines.append("**`<think>...</think>` shows up inline in content. "
                     "The reasoning parser is missing. Stop.**")
        hard_fail = True

    # Thinking-off request.
    lines.append("")
    lines.append("## Thinking OFF")
    try:
        resp_off, req_off = _make_request(client, thinking=False, user_msg=user_msg, system_prompt=sys_prompt)
        msg_off = resp_off.choices[0].message
        reasoning_off = extract_reasoning(msg_off)
        content_off = msg_off.content or ""
        content_off_valid = False
        try:
            ProverbTranslation.model_validate_json(content_off.strip())
            content_off_valid = True
        except Exception:
            pass
        lines.append(f"- content validates: {content_off_valid}")
        lines.append(f"- reasoning empty: {not bool(reasoning_off)} (len={len(reasoning_off)})")
    except Exception as e:
        lines.append(f"- thinking-off request failed: {e!r}")
        hard_fail = True

    # Record example outputs for human inspection.
    lines.append("")
    lines.append("## Example outputs")
    lines.append("### Thinking ON — content")
    lines.append("```")
    lines.append(content_on[:2000])
    lines.append("```")
    if reasoning_on:
        lines.append("### Thinking ON — reasoning (first 1,000 chars)")
        lines.append("```")
        lines.append(reasoning_on[:1000])
        lines.append("```")

    Path(config.RUNS / "preflight.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Persist what we learned for later steps. (Omit base_url: it's read from
    # $QWEN_BASE_URL at run time, and keeping it out of the JSON avoids
    # committing an internal host if the file ever gets added to git.)
    pre_json = {
        "vllm_version": vllm_version,
        "thinking_token_budget_accepted": budget_accepted,
        "reasoning_tokens_in_usage": reasoning_tok_present,
    }
    (config.RUNS / "preflight.json").write_text(json.dumps(pre_json, indent=2), encoding="utf-8")

    print(f"Preflight written to {config.RUNS / 'preflight.md'}", file=sys.stderr)
    return 1 if hard_fail else 0


def main() -> int:
    return run_preflight()


if __name__ == "__main__":
    raise SystemExit(main())
