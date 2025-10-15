from __future__ import annotations

import os
import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse


# ---- Configuration ----

# Bind address/port
HOST = "0.0.0.0"
PORT = 9999

# HTTP proxy for upstream requests (both http and https)
HTTP_PROXY = "http://127.0.0.1:7890"

# Debug logging toggle (set TAIDIGEST_DEBUG=0 to disable)
DEBUG_LOG = os.getenv("TAIDIGEST_DEBUG", "1") != "1"

# Fold history for Claude upstream to ensure multi-turn context even if
# upstream ignores assistant turns. Set TAIDIGEST_FOLD_HISTORY_FOR_CLAUDE=0 to disable.
FOLD_HISTORY_FOR_CLAUDE = os.getenv("TAIDIGEST_FOLD_HISTORY_FOR_CLAUDE", "1") != "0"

def debug_log(*args: Any, **kwargs: Any) -> None:
    if DEBUG_LOG:
        try:
            print(*args, **kwargs)
        except Exception:
            pass


# ---- Constants & Helpers ----

CORS_ALLOW_HEADERS = [
    "authorization",
    "content-type",
    "x-api-key",
    "anthropic-version",
]

OPENAI_MODELS = [
    "gpt-4o-2024-08-06",
    "gpt-4o-mini-2024-07-18",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
]

ANTHROPIC_MODELS = [
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-7-sonnet-20250219",
    "claude-sonnet-4-0",
    "claude-3-opus-20240229",
    "claude-opus-4-20250514",
]

ALL_MODELS = OPENAI_MODELS + ANTHROPIC_MODELS


def is_claude_model(model: str) -> bool:
    return model.lower().startswith("claude")


def text_only(content: Any) -> str:
    """Extract text from various content shapes.

    - string -> itself
    - list[{type: "text", text: str}] -> concatenated text
    - object with .content or .text -> respective string
    - otherwise -> empty string
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        out_parts: List[str] = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text" and isinstance(c.get("text"), str):
                out_parts.append(c["text"]) 
        return "".join(out_parts)

    if isinstance(content, dict):
        if isinstance(content.get("content"), str):
            return content["content"]
        if content.get("type") == "text" and isinstance(content.get("text"), str):
            return content["text"]

    return ""


# ---- Normalized Input Types ----

Role = str  # "system" | "user" | "assistant"


class NormalizedMessage(Dict[str, Any]):
    role: Role
    content: str


class NormalizedInput(Dict[str, Any]):
    model: str
    messages: List[NormalizedMessage]
    system: Optional[str]
    temperature: Optional[float]
    stream: bool


def try_from_anthropic(body: Any) -> Optional[NormalizedInput]:
    if not isinstance(body, dict):
        return None
    if not isinstance(body.get("messages"), list):
        return None
    if not body.get("model"):
        return None

    model = str(body["model"]) 
    messages: List[NormalizedMessage] = []

    for m in body.get("messages", []):
        role = str((m or {}).get("role", ""))
        if role not in ("system", "user", "assistant"):
            continue
        text = text_only((m or {}).get("content"))
        messages.append({"role": role, "content": text})

    system: Optional[str] = None
    if isinstance(body.get("system"), str):
        system = body["system"]
    else:
        sys_joined = "\n".join(x["content"] for x in messages if x["role"] == "system")
        if sys_joined:
            system = sys_joined

    temperature = body.get("temperature") if isinstance(body.get("temperature"), (int, float)) else None
    stream = bool(body.get("stream"))

    return NormalizedInput(
        model=model,
        messages=messages,
        system=system,
        temperature=float(temperature) if temperature is not None else None,
        stream=stream,
    )


def try_from_openai_chat(body: Any) -> Optional[NormalizedInput]:
    if not isinstance(body, dict):
        return None
    if not isinstance(body.get("messages"), list):
        return None
    if not body.get("model"):
        return None

    model = str(body["model"]) 
    messages: List[NormalizedMessage] = []
    for m in body.get("messages", []):
        role = str((m or {}).get("role", ""))
        if role not in ("system", "user", "assistant"):
            continue
        text = text_only((m or {}).get("content"))
        messages.append({"role": role, "content": text})

    system: Optional[str] = None
    sys_joined = "\n".join(x["content"] for x in messages if x["role"] == "system")
    if sys_joined:
        system = sys_joined

    temperature = body.get("temperature") if isinstance(body.get("temperature"), (int, float)) else None
    stream = bool(body.get("stream"))

    return NormalizedInput(
        model=model,
        messages=messages,
        system=system,
        temperature=float(temperature) if temperature is not None else None,
        stream=stream,
    )


def try_from_openai_responses(body: Any) -> Optional[NormalizedInput]:
    if not isinstance(body, dict):
        return None
    if not body.get("model"):
        return None

    model = str(body["model"]) 
    system = body.get("system") if isinstance(body.get("system"), str) else None
    user_text = text_only(body.get("input"))

    messages: List[NormalizedMessage] = []
    if system:
        messages.append({"role": "system", "content": system})
    if user_text:
        messages.append({"role": "user", "content": user_text})

    temperature = body.get("temperature") if isinstance(body.get("temperature"), (int, float)) else None
    stream = bool(body.get("stream"))

    return NormalizedInput(
        model=model,
        messages=messages,
        system=system,
        temperature=float(temperature) if temperature is not None else None,
        stream=stream,
    )


def normalize(body: Any) -> NormalizedInput:
    a = try_from_anthropic(body)
    if a:
        return a
    b = try_from_openai_chat(body)
    if b:
        return b
    c = try_from_openai_responses(body)
    if c:
        return c
    raise ValueError("Invalid body: cannot normalize from known schemas.")


# ---- Upstream shaping ----

def to_anthropic_body(n: NormalizedInput) -> Dict[str, Any]:
    system: Optional[str] = n.get("system")
    if not system:
        joined = "\n".join(m["content"] for m in n["messages"] if m["role"] == "system")
        if joined:
            system = joined

    def _fold_transcript(msgs: List[NormalizedMessage]) -> str:
        parts: List[str] = []
        for m in msgs:
            role = m.get("role")
            if role == "system":
                # Keep system separate via 'system' field to avoid duplication
                continue
            content = str(m.get("content") or "")
            label = "User" if role == "user" else "Assistant"
            parts.append(f"{label}: {content}")
        return "\n\n".join(parts)

    # If configured, fold history (user+assistant turns) into a single user message
    # to ensure upstream sees full context.
    use_fold = FOLD_HISTORY_FOR_CLAUDE and any(m.get("role") == "assistant" for m in n["messages"])
    if use_fold:
        transcript = _fold_transcript(n["messages"])
        msgs = [{
            "role": "user",
            "content": [{"type": "text", "text": transcript}],
        }]
    else:
        msgs = [
            {
                "role": m["role"],
                "content": [{"type": "text", "text": m["content"]}],
            }
            for m in n["messages"]
            if m["role"] in ("user", "assistant")
        ]

    body: Dict[str, Any] = {
        "model": n["model"],
        "messages": msgs,
        "stream": True,
    }
    if system:
        body["system"] = system
    if isinstance(n.get("temperature"), (int, float)):
        body["temperature"] = n["temperature"]
    return body


def to_openai_chat_body(n: NormalizedInput) -> Dict[str, Any]:
    # Use plain string content for maximum compatibility with Chat Completions
    messages = [
        {
            "role": m["role"],
            "content": m["content"],
        }
        for m in n["messages"]
    ]
    body: Dict[str, Any] = {
        "model": n["model"],
        "messages": messages,
        "stream": True,
    }
    if isinstance(n.get("temperature"), (int, float)):
        body["temperature"] = n["temperature"]
    return body


UPSTREAM_OPENAI = "https://theaidigest.org/agent/api/openai"
UPSTREAM_ANTHROPIC = "https://theaidigest.org/agent/api/anthropic"


def upstream_url_for(n: NormalizedInput) -> str:
    return UPSTREAM_ANTHROPIC if is_claude_model(n["model"]) else UPSTREAM_OPENAI


# Shared httpx client with proxy
_client: Optional[httpx.AsyncClient] = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            proxy=HTTP_PROXY,
            timeout=httpx.Timeout(60.0, read=300.0),  # generous read timeout for SSE
        )
    return _client


async def fetch_upstream_stream(n: NormalizedInput) -> httpx.Response:
    url = upstream_url_for(n)
    body = to_anthropic_body(n) if is_claude_model(n["model"]) else to_openai_chat_body(n)
    client = get_client()
    # Build and send a streaming request (avoid pre-buffering response)
    try:
        debug_log(f"[UPSTREAM] POST {url} body={json.dumps(body, ensure_ascii=False)}")
    except Exception:
        pass
    req = client.build_request("POST", url, json=body, headers={"content-type": "application/json"})
    resp = await client.send(req, stream=True)
    return resp


# ---- SSE helpers ----

async def iter_sse_data_lines_from_streaming_response(resp: httpx.Response) -> AsyncGenerator[str, None]:
    """Yield concatenated data-lines for each SSE event from a streaming response.

    Extracts lines that begin with 'data:' and concatenates them per event.
    Events are delimited by blank lines (\n\n).
    """
    if resp.is_closed:
        # In case caller accidentally passed a fully-read response
        return

    decoder = None
    try:
        import codecs

        decoder = codecs.getincrementaldecoder("utf-8")()
    except Exception:  # pragma: no cover
        decoder = None

    buf = ""
    async for chunk in resp.aiter_bytes():
        piece = chunk.decode("utf-8", errors="ignore") if decoder is None else decoder.decode(chunk)
        buf += piece

        while True:
            idx = buf.find("\n\n")
            if idx == -1:
                break
            chunk_str = buf[:idx]
            buf = buf[idx + 2 :]
            data_lines = [ln[5:].lstrip() for ln in chunk_str.split("\n") if ln.startswith("data:")]
            if data_lines:
                yield "\n".join(data_lines)

    # any trailing data
    trailing = buf.strip()
    if trailing:
        data_lines = [ln[5:].lstrip() for ln in trailing.split("\n") if ln.startswith("data:")]
        if data_lines:
            yield "\n".join(data_lines)


async def aggregate_from_upstream(n: NormalizedInput) -> Tuple[str, Optional[str]]:
    """Aggregate upstream (SSE or JSON) into a plain text and finish reason."""
    resp = await fetch_upstream_stream(n)
    try:
        if resp.status_code < 200 or resp.status_code >= 300:
            text = await resp.aread()
            raise RuntimeError(f"Upstream error: {resp.status_code} {text.decode(errors='ignore')}")

        content_type = resp.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            out = ""
            finish: Optional[str] = None
            async for data in iter_sse_data_lines_from_streaming_response(resp):
                if not data or data == "[DONE]":
                    continue
                try:
                    j = json.loads(data)
                except Exception:
                    continue

                # Non-standard SSE: {"data":"..."}
                if isinstance(j, dict) and isinstance(j.get("data"), str):
                    out += j["data"]

                # OpenAI-like SSE
                if isinstance(j, dict) and j.get("choices"):
                    delta = j["choices"][0].get("delta", {}) if j["choices"] else {}
                    if isinstance(delta.get("content"), str):
                        out += delta["content"]
                    if not finish and j["choices"][0].get("finish_reason"):
                        finish = j["choices"][0].get("finish_reason")

                # Anthropic-like SSE
                if j.get("type") == "content_block_delta" and j.get("delta", {}).get("type") == "text_delta":
                    t = j.get("delta", {}).get("text")
                    if isinstance(t, str):
                        out += t
                if not finish and j.get("type") == "message_delta" and j.get("delta", {}).get("stop_reason"):
                    finish = j.get("delta", {}).get("stop_reason")

            return out, finish

        # JSON fallback: fully read the response and decode
        try:
            raw = await resp.aread()
            j = json.loads(raw.decode("utf-8", errors="ignore"))
        except Exception:
            j = {}

        if isinstance(j, dict) and isinstance(j.get("content"), list):
            text = "".join(
                b.get("text", "") for b in j["content"] if isinstance(b, dict) and b.get("type") == "text"
            )
            stop = j.get("stop_reason") or j.get("finish_reason")
            return text, stop

        text = str(j.get("output_text") or j.get("text") or (j.get("message") or {}).get("content") or "")
        stop = j.get("stop_reason") or j.get("finish_reason")
        return text, stop
    finally:
        await resp.aclose()


def build_openai_chat_non_stream(model: str, text: str, finish: Optional[str] = None) -> Dict[str, Any]:
    now = int(time.time())
    return {
        "id": f"chatcmpl_{uuid.uuid4()}",
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish or "stop"},
        ],
    }


def build_anthropic_message(model: str, text: str, stop_reason: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": f"msg_{uuid.uuid4()}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason or "end_turn",
        "stop_sequence": None,
    }


# ---- FastAPI app ----

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=CORS_ALLOW_HEADERS + ["*"],
)


@app.on_event("shutdown")
async def _shutdown_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


# ---- Simple in-memory conversation store (optional) ----

# Store messages per conversation_id. Each value is a list[NormalizedMessage]
CONVERSATIONS: dict[str, List[NormalizedMessage]] = {}
CONV_LOCK = asyncio.Lock()
MAX_MESSAGES = 50  # cap stored messages per conversation


def _has_assistant_messages(msgs: List[NormalizedMessage]) -> bool:
    return any(m.get("role") == "assistant" for m in msgs)


def _only_single_user_turn(msgs: List[NormalizedMessage]) -> bool:
    non_system = [m for m in msgs if m.get("role") != "system"]
    return len(non_system) == 1 and non_system[0].get("role") == "user"


def _merge_system(existing: List[NormalizedMessage], incoming: List[NormalizedMessage]) -> List[NormalizedMessage]:
    # Keep a single system message at the start. Prefer incoming system if present, else existing.
    sys_incoming = [m for m in incoming if m.get("role") == "system"]
    sys_existing = [m for m in existing if m.get("role") == "system"]
    sys_final: List[NormalizedMessage] = []
    if sys_incoming:
        sys_final = sys_incoming[:1]
    elif sys_existing:
        sys_final = sys_existing[:1]

    # Now collect non-system messages
    existing_ns = [m for m in existing if m.get("role") in ("user", "assistant")]
    incoming_ns = [m for m in incoming if m.get("role") in ("user", "assistant")]
    return sys_final + existing_ns + incoming_ns


def _trim_messages(msgs: List[NormalizedMessage]) -> List[NormalizedMessage]:
    # Ensure we don't exceed MAX_MESSAGES by trimming from the front, but keep system if present
    system_msgs = [m for m in msgs if m.get("role") == "system"]
    rest = [m for m in msgs if m.get("role") != "system"]
    if len(rest) > MAX_MESSAGES:
        rest = rest[-MAX_MESSAGES:]
    return (system_msgs[:1] + rest) if system_msgs else rest


async def _upsert_system(conv_id: str, msgs: List[NormalizedMessage]) -> None:
    if not conv_id:
        return
    sys_incoming = next((m for m in msgs if m.get("role") == "system" and isinstance(m.get("content"), str) and m.get("content")), None)
    if not sys_incoming:
        return
    async with CONV_LOCK:
        cur = CONVERSATIONS.get(conv_id, [])
        # Remove any existing system entries
        cur = [m for m in cur if m.get("role") != "system"]
        # Prepend the new system message
        cur = [sys_incoming] + cur
        CONVERSATIONS[conv_id] = _trim_messages(cur)


def _get_conv_id_from_request_body(body: dict, req: Optional[Request] = None) -> Optional[str]:
    # Support: body.conversation_id, body.conversation (string), or header X-Conversation-Id
    cid = None
    if isinstance(body.get("conversation_id"), str) and body["conversation_id"].strip():
        cid = body["conversation_id"].strip()
    elif isinstance(body.get("conversation"), str) and body["conversation"].strip():
        cid = body["conversation"].strip()
    elif isinstance(body.get("user"), str) and body["user"].strip():
        # Fallback: use OpenAI 'user' field as a conversation key if provided.
        # Note: this may group multiple threads per user into one, so prefer explicit conversation_id.
        cid = body["user"].strip()
    if not cid and req is not None:
        h = req.headers.get("x-conversation-id") or req.headers.get("X-Conversation-Id")
        if h and h.strip():
            cid = h.strip()
    return cid


async def _maybe_attach_history(norm: NormalizedInput, body: dict, req: Optional[Request]) -> Tuple[NormalizedInput, Optional[str], Optional[NormalizedMessage]]:
    """If a conversation_id is provided and incoming lacks history, prepend stored history.

    Returns: (possibly-updated norm, conversation_id, incoming_user_message)
    """
    conv_id = _get_conv_id_from_request_body(body, req)
    if not conv_id:
        return norm, None, None

    msgs = norm["messages"]
    # Heuristic: only attach history if incoming has at most a single user turn and no assistant messages
    if _has_assistant_messages(msgs) or not _only_single_user_turn(msgs):
        return norm, conv_id, None

    incoming_user = next((m for m in msgs if m.get("role") == "user"), None)
    if not incoming_user:
        return norm, conv_id, None

    async with CONV_LOCK:
        history = CONVERSATIONS.get(conv_id, [])
    if not history:
        return norm, conv_id, incoming_user

    merged = _merge_system(history, msgs)
    norm_with_history = NormalizedInput(
        model=norm["model"],
        messages=merged,
        system=norm.get("system"),
        temperature=norm.get("temperature"),
        stream=norm.get("stream", False),
    )
    return norm_with_history, conv_id, incoming_user


async def _save_turn(conv_id: str, user_msg: NormalizedMessage, assistant_text: str) -> None:
    if not conv_id:
        return
    user_entry = {"role": "user", "content": user_msg.get("content", "")}
    assistant_entry = {"role": "assistant", "content": assistant_text}
    async with CONV_LOCK:
        cur = CONVERSATIONS.get(conv_id, [])
        # If no system in cur but present in user_msg? We only store user/assistant here; system handled by merge logic
        cur = cur + [user_entry, assistant_entry]
        CONVERSATIONS[conv_id] = _trim_messages(cur)


@app.get("/v1/models")
async def get_models() -> JSONResponse:
    payload = {
        "object": "list",
        "data": [
            {"id": m, "provider": "anthropic" if is_claude_model(m) else "openai"} for m in ALL_MODELS
        ],
    }
    return JSONResponse(payload)


async def stream_as_openai_chunks(n: NormalizedInput) -> StreamingResponse:
    resp = await fetch_upstream_stream(n)
    if resp.status_code < 200 or resp.status_code >= 300:
        text = await resp.aread()
        await resp.aclose()
        return JSONResponse({"error": f"Upstream error: {resp.status_code} {text.decode(errors='ignore')}"}, status_code=500)

    created = int(time.time())
    content_type = resp.headers.get("content-type", "")

    async def event_generator() -> AsyncGenerator[bytes, None]:
        enc = lambda obj: f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")

        # If upstream is not SSE, aggregate then send one final chunk
        if "text/event-stream" not in content_type:
            text, _finish = await aggregate_from_upstream(n)
            one = {
                "id": f"chatcmpl_{uuid.uuid4()}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": n["model"],
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            }
            yield enc(one)
            yield b"data: [DONE]\n\n"
            return

        sent_role = False
        async for data in iter_sse_data_lines_from_streaming_response(resp):
            if not data or data == "[DONE]":
                yield b"data: [DONE]\n\n"
                break

            try:
                j = json.loads(data)
            except Exception:
                continue

            # Non-standard: {"data": "..."}
            if isinstance(j, dict) and isinstance(j.get("data"), str):
                delta: Dict[str, Any] = {"content": j["data"]}
                if not sent_role:
                    delta["role"] = "assistant"
                    sent_role = True
                out = {
                    "id": f"chatcmpl_{uuid.uuid4()}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": n["model"],
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                yield enc(out)

            # OpenAI-like passthrough
            if isinstance(j, dict) and j.get("choices"):
                yield enc(j)
                if (j.get("choices", [{}])[0] or {}).get("finish_reason"):
                    yield b"data: [DONE]\n\n"
                    break

            # Anthropic -> OpenAI delta
            if j.get("type") == "content_block_delta" and j.get("delta", {}).get("type") == "text_delta" and isinstance(j.get("delta", {}).get("text"), str):
                delta: Dict[str, Any] = {"content": j["delta"]["text"]}
                if not sent_role:
                    delta["role"] = "assistant"
                    sent_role = True
                out = {
                    "id": f"chatcmpl_{uuid.uuid4()}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": n["model"],
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                yield enc(out)

            if j.get("type") == "message_delta" and j.get("delta", {}).get("stop_reason"):
                end = {
                    "id": f"chatcmpl_{uuid.uuid4()}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": n["model"],
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield enc(end)
                yield b"data: [DONE]\n\n"
                break

        # Safety DONE in case upstream ends without explicit finish
        yield b"data: [DONE]\n\n"
        await resp.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "cache-control": "no-cache, no-transform",
            "connection": "keep-alive",
            "x-accel-buffering": "no",
        },
    )


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    # Log raw request and parse JSON
    try:
        raw = await req.body()
        raw_text = raw.decode("utf-8", errors="ignore")
        debug_log(f"[REQ] {req.method} {req.url.path} X-Conversation-Id={req.headers.get('x-conversation-id')} body={raw_text}")
        try:
            body = json.loads(raw_text) if raw_text else {}
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    try:
        norm = try_from_openai_chat(body) or normalize(body)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # Optional memory: attach history if conversation_id provided and incoming lacks history
    norm2, conv_id, incoming_user = await _maybe_attach_history(norm, body, req)
    if conv_id:
        debug_log(f"[MEM] conversation_id={conv_id} attach_history={'yes' if incoming_user else 'no'}")
        await _upsert_system(conv_id, norm.get("messages", []))
    if conv_id:
        await _upsert_system(conv_id, norm.get("messages", []))

    try:
        if norm2["stream"]:
            # Stream, but also accumulate assistant text to save at the end
            resp = await fetch_upstream_stream(norm2)
            if resp.status_code < 200 or resp.status_code >= 300:
                text = await resp.aread()
                await resp.aclose()
                return JSONResponse({"error": f"Upstream error: {resp.status_code} {text.decode(errors='ignore')}"}, status_code=500)

            created = int(time.time())
            content_type = resp.headers.get("content-type", "")
            assistant_buf: List[str] = []

            async def event_generator():
                enc = lambda obj: f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")

                # If upstream is not SSE, aggregate and send one chunk
                if "text/event-stream" not in content_type:
                    text, _finish = await aggregate_from_upstream(norm2)
                    one = {
                        "id": f"chatcmpl_{uuid.uuid4()}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": norm2["model"],
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                    }
                    assistant_buf.append(text)
                    yield enc(one)
                    yield b"data: [DONE]\n\n"
                    await resp.aclose()
                    # Save memory after finish
                    if conv_id and incoming_user and assistant_buf:
                        await _save_turn(conv_id, incoming_user, "".join(assistant_buf))
                    return

                sent_role = False
                async for data in iter_sse_data_lines_from_streaming_response(resp):
                    if not data or data == "[DONE]":
                        yield b"data: [DONE]\n\n"
                        break

                    try:
                        j = json.loads(data)
                    except Exception:
                        continue

                    # Non-standard: {"data": "..."}
                    if isinstance(j, dict) and isinstance(j.get("data"), str):
                        assistant_buf.append(j["data"])
                        delta = {"content": j["data"]}
                        if not sent_role:
                            delta["role"] = "assistant"
                            sent_role = True
                        out = {
                            "id": f"chatcmpl_{uuid.uuid4()}",
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": norm2["model"],
                            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                        }
                        yield enc(out)

                    # OpenAI-like passthrough
                    if isinstance(j, dict) and j.get("choices"):
                        delta = (j.get("choices", [{}])[0] or {}).get("delta", {})
                        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                            assistant_buf.append(delta["content"])
                        yield enc(j)
                        if (j.get("choices", [{}])[0] or {}).get("finish_reason"):
                            yield b"data: [DONE]\n\n"
                            break

                    # Anthropic -> OpenAI delta
                    if j.get("type") == "content_block_delta" and j.get("delta", {}).get("type") == "text_delta" and isinstance(j.get("delta", {}).get("text"), str):
                        assistant_buf.append(j["delta"]["text"])
                        delta = {"content": j["delta"]["text"]}
                        if not sent_role:
                            delta["role"] = "assistant"
                            sent_role = True
                        out = {
                            "id": f"chatcmpl_{uuid.uuid4()}",
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": norm2["model"],
                            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                        }
                        yield enc(out)

                    if j.get("type") == "message_delta" and j.get("delta", {}).get("stop_reason"):
                        end = {
                            "id": f"chatcmpl_{uuid.uuid4()}",
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": norm2["model"],
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }
                        yield enc(end)
                        yield b"data: [DONE]\n\n"
                        break

                # Safety DONE
                yield b"data: [DONE]\n\n"
                await resp.aclose()
                # Save memory after finish
                if conv_id and incoming_user and assistant_buf:
                    await _save_turn(conv_id, incoming_user, "".join(assistant_buf))

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "cache-control": "no-cache, no-transform",
                    "connection": "keep-alive",
                    "x-accel-buffering": "no",
                },
            )
        else:
            text, finish = await aggregate_from_upstream(norm2)
            # Save to memory for non-stream
            if conv_id and incoming_user:
                await _save_turn(conv_id, incoming_user, text)
            return JSONResponse(build_openai_chat_non_stream(norm2["model"], text, finish))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/v1/responses")
async def responses(req: Request):
    # Log raw request and parse JSON
    try:
        raw = await req.body()
        raw_text = raw.decode("utf-8", errors="ignore")
        debug_log(f"[REQ] {req.method} {req.url.path} X-Conversation-Id={req.headers.get('x-conversation-id')} body={raw_text}")
        try:
            body = json.loads(raw_text) if raw_text else {}
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    try:
        norm = try_from_openai_responses(body) or normalize(body)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    try:
        # Optional memory
        norm2, conv_id, incoming_user = await _maybe_attach_history(norm, body, req)
        if conv_id:
            debug_log(f"[MEM] conversation_id={conv_id} attach_history={'yes' if incoming_user else 'no'}")
            await _upsert_system(conv_id, norm.get("messages", []))
        text, _finish = await aggregate_from_upstream(norm2)
        if conv_id and incoming_user:
            await _save_turn(conv_id, incoming_user, text)
        out = {
            "id": f"resp_{uuid.uuid4()}",
            "object": "response",
            "model": norm2["model"],
            "created": int(time.time()),
            "output": [{"type": "output_text", "text": text}],
            "output_text": text,
        }
        return JSONResponse(out)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/v1/messages")
async def anthropic_messages(req: Request):
    # Log raw request and parse JSON
    try:
        raw = await req.body()
        raw_text = raw.decode("utf-8", errors="ignore")
        debug_log(f"[REQ] {req.method} {req.url.path} X-Conversation-Id={req.headers.get('x-conversation-id')} body={raw_text}")
        try:
            body = json.loads(raw_text) if raw_text else {}
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    try:
        norm = try_from_anthropic(body) or normalize(body)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    try:
        # Optional memory
        norm2, conv_id, incoming_user = await _maybe_attach_history(norm, body, req)
        if conv_id:
            debug_log(f"[MEM] conversation_id={conv_id} attach_history={'yes' if incoming_user else 'no'}")
            await _upsert_system(conv_id, norm.get("messages", []))
        text, finish = await aggregate_from_upstream(norm2)
        if conv_id and incoming_user:
            await _save_turn(conv_id, incoming_user, text)
        return JSONResponse(build_anthropic_message(norm2["model"], text, finish))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({"ok": True, "service": "theaidigiest2API proxy", "endpoints": [
        "/v1/models", "/v1/chat/completions", "/v1/responses", "/v1/messages"
    ]})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
