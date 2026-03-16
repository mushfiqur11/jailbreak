"""Robust JSON parsing helpers for agent prompt outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ParseResult:
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: str = ""
    raw_response: str = ""


def _strip_reasoning_blocks(text: str) -> str:
    """Remove common reasoning wrappers (e.g., <think>...</think>) from model output."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


def _extract_fenced_blocks(text: str) -> List[str]:
    """Extract content from fenced code blocks, preferring ```json fences."""
    blocks: List[str] = []
    for m in re.finditer(r"```json\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE):
        blocks.append(m.group(1).strip())
    for m in re.finditer(r"```\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE):
        b = m.group(1).strip()
        if b and b not in blocks:
            blocks.append(b)
    return blocks


def _find_balanced_json_objects(text: str) -> List[str]:
    """Extract all balanced {...} object substrings, robust to quoted braces."""
    candidates: List[str] = []
    n = len(text)
    i = 0

    while i < n:
        if text[i] != "{":
            i += 1
            continue

        start = i
        depth = 0
        in_str = False
        esc = False

        j = i
        while j < n:
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start:j + 1])
                        break
            j += 1

        i = j + 1

    return candidates


def _extract_json_candidates(response: str) -> List[str]:
    text = _strip_reasoning_blocks(response.strip())
    if not text:
        return []

    candidates: List[str] = []

    # Prefer explicitly fenced JSON blocks when present.
    # Extract whole fence content first, then locate balanced objects within it.
    for block in _extract_fenced_blocks(text):
        block_objects = _find_balanced_json_objects(block)
        if block_objects:
            candidates.extend(block_objects)
        else:
            # Keep raw block as candidate (may be slightly malformed but recoverable)
            candidates.append(block)

    # Also collect balanced objects from full output (handles extra prose)
    candidates.extend(_find_balanced_json_objects(text))

    # Last-resort fallback: take suffix from first '{' if output is truncated/unbalanced
    first_brace = text.find("{")
    if first_brace != -1:
        candidates.append(text[first_brace:].strip())

    # Deduplicate while preserving order
    seen = set()
    unique_candidates: List[str] = []
    for c in candidates:
        key = c.strip()
        if key and key not in seen:
            seen.add(key)
            unique_candidates.append(key)

    return unique_candidates


def _repair_common_json_issues(payload: str) -> List[str]:
    """Generate common repaired variants for slightly malformed JSON payloads."""
    variants: List[str] = []
    s = payload.strip()
    if not s:
        return variants

    variants.append(s)

    # Handle occasional double-wrap: {{ ... }}
    if s.startswith("{{") and s.endswith("}}"):
        variants.append(s[1:-1].strip())

    # Convert single quotes variant
    variants.append(re.sub(r"(?<!\\)'", '"', s))

    # Brace-balance recovery for truncated outputs
    opens = s.count("{")
    closes = s.count("}")
    if opens > closes:
        variants.append(s + ("}" * (opens - closes)))

    # Deduplicate preserving order
    out: List[str] = []
    seen = set()
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def parse_llm_response_to_dict(response: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    if not isinstance(response, str):
        return None, "Invalid input: expected string response"

    stripped = response.strip()
    if not stripped:
        return None, "Received empty response after stripping"

    candidates = _extract_json_candidates(stripped)
    if not candidates:
        return None, "No dictionary structure detected in response"

    # Try candidates in order: fenced JSON candidates are intentionally added first.
    for candidate in candidates:
        for payload in _repair_common_json_issues(candidate):
            try:
                obj = json.loads(payload)
                if isinstance(obj, dict):
                    return obj, ""
            except Exception:
                continue

    return None, "Failed to parse dictionary JSON after handling common formatting issues"


def validate_required_fields(
    data: Any,
    required_fields: List[str],
    field_types: Optional[Dict[str, type]] = None,
) -> Tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "Input is not a dictionary"

    missing = [f for f in required_fields if f not in data]
    if missing:
        return False, f"Missing required fields: {', '.join(missing)}"

    if field_types:
        errs = []
        for f, t in field_types.items():
            if f in data and data[f] is not None and not isinstance(data[f], t):
                errs.append(f"{f} expected {getattr(t, '__name__', str(t))}, got {type(data[f]).__name__}")
        if errs:
            return False, "Type validation errors: " + "; ".join(errs)

    return True, ""


def safe_parse_with_validation(
    response: str,
    required_fields: List[str],
    field_types: Optional[Dict[str, type]] = None,
    parser_name: str = "",
) -> ParseResult:
    try:
        data, err = parse_llm_response_to_dict(response)
        if data is None:
            return ParseResult(False, None, f"JSON parsing failed{f' in {parser_name}' if parser_name else ''}: {err}", response)

        ok, verr = validate_required_fields(data, required_fields, field_types)
        # Recovery path: model sometimes wraps target object as JSON string in a top-level "response" field.
        if (not ok) and isinstance(data.get("response"), str):
            nested_data, nested_err = parse_llm_response_to_dict(data["response"])
            if nested_data is not None:
                ok, verr = validate_required_fields(nested_data, required_fields, field_types)
                if ok:
                    return ParseResult(True, nested_data, "", response)
            else:
                # Keep original validation error but include nested parse hint for debugging.
                verr = f"{verr}; nested response parse failed: {nested_err}"

        if not ok:
            return ParseResult(False, data, f"Validation failed{f' in {parser_name}' if parser_name else ''}: {verr}", response)

        return ParseResult(True, data, "", response)
    except Exception as e:
        return ParseResult(False, None, f"Unexpected parsing error{f' in {parser_name}' if parser_name else ''}: {e}", response)
