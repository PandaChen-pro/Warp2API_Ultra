// main.ts
// Run: deno run -A main.ts
// 无需上游 key；根据模型自动选择上游：GPT -> https://theaidigest.org/agent/api/openai，Claude -> https://theaidigest.org/agent/api/anthropic
// 端点：/v1/chat/completions, /v1/responses, /v1/messages, /v1/models
// 行为：强制上游 stream；前端 stream=true 则下游 SSE(OpenAI chunk)；否则聚合为非流 JSON 返回[4]

type Role = "system" | "user" | "assistant";

type NormalizedMessage = {
  role: Role;
  content: string;
};

type NormalizedInput = {
  model: string;
  messages: NormalizedMessage[]; 
  system?: string;
  temperature?: number;
  stream?: boolean;
};

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization,content-type,x-api-key,anthropic-version",
  "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
};

const OPENAI_MODELS = [
  "gpt-4o-2024-08-06",
  "gpt-4o-mini-2024-07-18",
  "gpt-4.1-2025-04-14",
  "gpt-4.1-mini-2025-04-14"
];

const ANTHROPIC_MODELS = [
  "claude-3-5-sonnet-20241022",
  "claude-3-5-sonnet-20240620",
  "claude-3-7-sonnet-20250219",
  "claude-sonnet-4-0",
  "claude-3-opus-20240229",
  "claude-opus-4-20250514"
];

const ALL_MODELS = [...OPENAI_MODELS, ...ANTHROPIC_MODELS];

function isClaudeModel(model: string): boolean {
  return model.toLowerCase().startsWith("claude");
}

function jsonHeaders(extra?: Record<string, string>) {
  return { "content-type": "application/json; charset=utf-8", ...CORS_HEADERS, ...(extra ?? {}) };
}

function okJSON(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200, headers: jsonHeaders() });
}

function badRequest(msg: string) {
  return new Response(JSON.stringify({ error: msg }), { status: 400, headers: jsonHeaders() });
}

function serverError(msg: string) {
  return new Response(JSON.stringify({ error: msg }), { status: 500, headers: jsonHeaders() });
}

function textOnly(content: any): string {
  if (typeof content === "string") return content;

  if (Array.isArray(content)) {
    return content
      .map((c) => (c && typeof c === "object" && c.type === "text" && typeof c.text === "string" ? c.text : ""))
      .join("");
  }

  if (content && typeof content === "object") {
    if (typeof content.content === "string") return content.content;
    if (content.type === "text" && typeof content.text === "string") return content.text;
  }

  return "";
}

// ---- Normalizers (Anthropic -> OpenAI Chat -> OpenAI Responses) ----

function tryFromAnthropic(body: any): NormalizedInput | null {
  if (!body || typeof body !== "object") return null;
  if (!Array.isArray(body.messages)) return null;
  if (!body.model) return null;

  const model = String(body.model);
  const messages: NormalizedMessage[] = [];

  for (const m of body.messages) {
    const role = String(m?.role ?? "");
    if (!["system", "user", "assistant"].includes(role)) continue;
    const text = textOnly(m?.content);
    messages.push({ role: role as Role, content: text });
  }

  let system: string | undefined;
  if (typeof body.system === "string") {
    system = body.system;
  } else {
    const sysJoined = messages
      .filter((x) => x.role === "system")
      .map((x) => x.content)
      .join("\n");
    if (sysJoined.length > 0) {
      system = sysJoined;
    }
  }

  const temperature = typeof body.temperature === "number" ? body.temperature : undefined;
  const stream = !!body.stream;

  return { model, messages, system, temperature, stream };
}

function tryFromOpenAIChat(body: any): NormalizedInput | null {
  if (!body || typeof body !== "object") return null;
  if (!Array.isArray(body.messages)) return null;
  if (!body.model) return null;

  const model = String(body.model);
  const messages: NormalizedMessage[] = [];

  for (const m of body.messages) {
    const role = String(m?.role ?? "");
    if (!["system", "user", "assistant"].includes(role)) continue;
    const text = textOnly(m?.content);
    messages.push({ role: role as Role, content: text });
  }

  let system: string | undefined;
  const sysJoined = messages
    .filter((x) => x.role === "system")
    .map((x) => x.content)
    .join("\n");
  if (sysJoined.length > 0) {
    system = sysJoined;
  }

  const temperature = typeof body.temperature === "number" ? body.temperature : undefined;
  const stream = !!body.stream;

  return { model, messages, system, temperature, stream };
}

function tryFromOpenAIResponses(body: any): NormalizedInput | null {
  if (!body || typeof body !== "object") return null;
  if (!body.model) return null;

  const model = String(body.model);
  const system = typeof body.system === "string" ? body.system : undefined;
  const userText = textOnly(body.input);

  const messages: NormalizedMessage[] = [];
  if (system) messages.push({ role: "system", content: system });
  if (userText) messages.push({ role: "user", content: userText });

  const temperature = typeof body.temperature === "number" ? body.temperature : undefined;
  const stream = !!body.stream;

  return { model, messages, system, temperature, stream };
}

function normalize(body: any): NormalizedInput {
  const a = tryFromAnthropic(body);
  if (a) return a;

  const b = tryFromOpenAIChat(body);
  if (b) return b;

  const c = tryFromOpenAIResponses(body);
  if (c) return c;

  throw new Error("Invalid body: cannot normalize from known schemas.");
}

// ---- SSE helpers ----

async function* sseDataLines(stream: ReadableStream<Uint8Array>) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buf += decoder.decode(value, { stream: true });

    let idx: number;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const chunk = buf.slice(0, idx);
      buf = buf.slice(idx + 2);

      const dataLines = chunk
        .split("\n")
        .filter((ln) => ln.startsWith("data:"))
        .map((ln) => ln.slice(5).trimStart());

      if (dataLines.length) {
        yield dataLines.join("\n");
      }
    }
  }

  if (buf.trim().length) {
    const dataLines = buf
      .split("\n")
      .filter((ln) => ln.startsWith("data:"))
      .map((ln) => ln.slice(5).trimStart());
    if (dataLines.length) yield dataLines.join("\n");
  }
}

// ---- Upstream selection & bodies (force stream upstream) ----

function toAnthropicBody(n: NormalizedInput) {
  let system: string | undefined = n.system;

  if (!system) {
    const joined = n.messages
      .filter((m) => m.role === "system")
      .map((m) => m.content)
      .join("\n");

    if (joined.length > 0) {
      system = joined;
    }
  }

  const msgs = n.messages
    .filter((m) => m.role === "user" || m.role === "assistant")
    .map((m) => ({
      role: m.role,
      content: [{ type: "text", text: m.content }],
    }));

  const body: any = {
    model: n.model,
    messages: msgs,
    stream: true,
  };

  if (system) body.system = system;
  if (typeof n.temperature === "number") body.temperature = n.temperature;

  return body;
}

function toOpenAIChatBody(n: NormalizedInput) {
  const messages = n.messages.map((m) => ({
    role: m.role,
    content: [
      { type: "text", text: m.content }
    ],
  }));
  const body: any = {
    model: n.model,
    messages,
    stream: true,
  };
  if (typeof n.temperature === "number") body.temperature = n.temperature;
  return body;
}


async function fetchUpstream(n: NormalizedInput): Promise<Response> {
  const url = isClaudeModel(n.model)
    ? "https://theaidigest.org/agent/api/anthropic"
    : "https://theaidigest.org/agent/api/openai";

  const body = isClaudeModel(n.model) ? toAnthropicBody(n) : toOpenAIChatBody(n);
    console.log(body)
    console.log(url)
  const resp = await fetch(url, {
    method: "POST",
    headers: {
      "content-type": "application/json",
    //   "accept": "text/event-stream",
    },
    body: JSON.stringify(body),
  });

  return resp;
}

// Aggregate any upstream (SSE or JSON) into plain text
async function aggregateFromUpstream(n: NormalizedInput): Promise<{ text: string; finish?: string }> {
  const resp = await fetchUpstream(n);

  if (!resp.ok) {
    const t = await resp.text().catch(() => "");
    throw new Error(`Upstream error: ${resp.status} ${t}`);
  }

  const contentType = resp.headers.get("content-type") || "";

  if (contentType.includes("text/event-stream") && resp.body) {
    let out = "";
    let finish: string | undefined;

    for await (const data of sseDataLines(resp.body)) {
      if (!data || data === "[DONE]") continue;

      try {
        const j = JSON.parse(data);

        // Non-standard SSE: data: {"data":"..."} -> append text
        if (typeof j?.data === "string") {
          out += j.data; // 手动补全文本片段[2]
        }

        // OpenAI-like SSE
        if (j?.choices?.[0]?.delta?.content) {
          out += j.choices[0].delta.content;
        }
        if (j?.choices?.[0]?.finish_reason && !finish) {
          finish = j.choices[0].finish_reason;
        }

        // Anthropic-like SSE
        if (j?.type === "content_block_delta" && j?.delta?.type === "text_delta" && typeof j?.delta?.text === "string") {
          out += j.delta.text;
        }
        if (j?.type === "message_delta" && j?.delta?.stop_reason && !finish) {
          finish = j.delta.stop_reason;
        }
      } catch {
        // ignore parse errors
      }
    }

    return { text: out, finish };
  }

  // JSON fallback
  const j = await resp.json().catch(() => ({} as any));

  if (Array.isArray(j?.content)) {
    const text = j.content
      .filter((b: any) => b && b.type === "text" && typeof b.text === "string")
      .map((b: any) => b.text)
      .join("");
    const stop = j.stop_reason ?? j.finish_reason ?? undefined;
    return { text, finish: stop };
  }

  const text = String(j?.output_text ?? j?.text ?? j?.message?.content ?? "");
  const stop = j?.stop_reason ?? j?.finish_reason ?? undefined;

  return { text, finish: stop };
}

// ---- Downstream SSE (OpenAI chat.completion.chunk) ----

function sseResponse(write: (ctrl: ReadableStreamDefaultController<Uint8Array>) => Promise<void>) {
  const enc = new TextEncoder();

  const stream = new ReadableStream<Uint8Array>({
    start: async (ctrl) => {
      try {
        await write(ctrl);
      } catch (e) {
        ctrl.enqueue(enc.encode(`data: ${JSON.stringify({ error: String(e) })}\n\n`));
      } finally {
        ctrl.close();
      }
    },
  });

  return new Response(stream, {
    status: 200,
    headers: {
      ...CORS_HEADERS,
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache, no-transform",
      "connection": "keep-alive",
      "x-accel-buffering": "no",
    },
  });
}

async function streamAsOpenAIChunks(n: NormalizedInput): Promise<Response> {
  const resp = await fetchUpstream(n);

  if (!resp.ok) {
    const t = await resp.text().catch(() => "");
    return serverError(`Upstream error: ${resp.status} ${t}`);
  }

  const created = Math.floor(Date.now() / 1000);
  const enc = new TextEncoder();
  const contentType = resp.headers.get("content-type") || "";

  return sseResponse(async (ctrl) => {
    let sentRole = false;

    if (!contentType.includes("text/event-stream") || !resp.body) {
      // Upstream non-SSE: aggregate then send one final chunk
      const { text } = await aggregateFromUpstream(n);
      const one = {
        id: `chatcmpl_${crypto.randomUUID()}`,
        object: "chat.completion.chunk",
        created,
        model: n.model,
        choices: [{ index: 0, delta: { role: "assistant", content: text }, finish_reason: "stop" }],
      };
      ctrl.enqueue(enc.encode(`data: ${JSON.stringify(one)}\n\n`));
      ctrl.enqueue(enc.encode("data: [DONE]\n\n"));
      return;
    }

    for await (const data of sseDataLines(resp.body)) {
      if (!data || data === "[DONE]") {
        ctrl.enqueue(enc.encode("data: [DONE]\n\n"));
        break;
      }

      try {
        const j = JSON.parse(data);

        // Non-standard SSE: {"data":"..."} -> map to delta.content
        if (typeof j?.data === "string") {
          const delta: any = { content: j.data };
          if (!sentRole) {
            delta.role = "assistant";
            sentRole = true;
          }
          const out = {
            id: `chatcmpl_${crypto.randomUUID()}`,
            object: "chat.completion.chunk",
            created,
            model: n.model,
            choices: [{ index: 0, delta, finish_reason: null }],
          };
          ctrl.enqueue(enc.encode(`data: ${JSON.stringify(out)}\n\n`));
        }

        // OpenAI-like chunk passthrough
        if (j?.choices && Array.isArray(j.choices)) {
          ctrl.enqueue(enc.encode(`data: ${JSON.stringify(j)}\n\n`));
          if (j?.choices?.[0]?.finish_reason) {
            ctrl.enqueue(enc.encode("data: [DONE]\n\n"));
            break;
          }
        }

        // Anthropic -> OpenAI delta
        if (j?.type === "content_block_delta" && j?.delta?.type === "text_delta" && typeof j?.delta?.text === "string") {
          const delta: any = { content: j.delta.text };
          if (!sentRole) {
            delta.role = "assistant";
            sentRole = true;
          }
          const out = {
            id: `chatcmpl_${crypto.randomUUID()}`,
            object: "chat.completion.chunk",
            created,
            model: n.model,
            choices: [{ index: 0, delta, finish_reason: null }],
          };
          ctrl.enqueue(enc.encode(`data: ${JSON.stringify(out)}\n\n`));
        }

        if (j?.type === "message_delta" && j?.delta?.stop_reason) {
          const end = {
            id: `chatcmpl_${crypto.randomUUID()}`,
            object: "chat.completion.chunk",
            created,
            model: n.model,
            choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
          };
          ctrl.enqueue(enc.encode(`data: ${JSON.stringify(end)}\n\n`));
          ctrl.enqueue(enc.encode("data: [DONE]\n\n"));
          break;
        }
      } catch {
        // ignore
      }
    }

    // Safety DONE in case upstream ends without explicit finish
    ctrl.enqueue(enc.encode("data: [DONE]\n\n"));
  });
}

// ---- Builders ----

function buildOpenAIChatNonStream(model: string, text: string, finish?: string) {
  const now = Math.floor(Date.now() / 1000);
  return {
    id: `chatcmpl_${crypto.randomUUID()}`,
    object: "chat.completion",
    created: now,
    model,
    choices: [
      { index: 0, message: { role: "assistant", content: text }, finish_reason: finish ?? "stop" },
    ],
  };
}

function buildAnthropicMessage(model: string, text: string, stop_reason?: string) {
  return {
    id: `msg_${crypto.randomUUID()}`,
    type: "message",
    role: "assistant",
    model,
    content: [{ type: "text", text }],
    stop_reason: stop_reason ?? "end_turn",
    stop_sequence: null,
  };
}

// ---- Routes ----

function modelsPayload() {
  return {
    object: "list",
    data: ALL_MODELS.map((id) => ({
      id,
      provider: isClaudeModel(id) ? "anthropic" : "openai",
    })),
  };
}

async function handleChatCompletions(req: Request) {
  let body: any;
  try {
    body = await req.json();
  } catch {
    return badRequest("Invalid JSON");
  }

  let norm: NormalizedInput;
  try {
    norm = tryFromOpenAIChat(body) ?? normalize(body);
  } catch (e) {
    return badRequest((e as Error).message);
  }

  try {
    if (norm.stream) {
      return await streamAsOpenAIChunks(norm);
    } else {
      const { text, finish } = await aggregateFromUpstream(norm); // 上游强制stream，本地聚合为非流[4]
      return okJSON(buildOpenAIChatNonStream(norm.model, text, finish));
    }
  } catch (e) {
    return serverError((e as Error).message);
  }
}

async function handleResponses(req: Request) {
  let body: any;
  try {
    body = await req.json();
  } catch {
    return badRequest("Invalid JSON");
  }

  let norm: NormalizedInput;
  try {
    norm = tryFromOpenAIResponses(body) ?? normalize(body);
  } catch (e) {
    return badRequest((e as Error).message);
  }

  try {
    const { text } = await aggregateFromUpstream(norm);
    const out = {
      id: `resp_${crypto.randomUUID()}`,
      object: "response",
      model: norm.model,
      created: Math.floor(Date.now() / 1000),
      output: [{ type: "output_text", text }],
      output_text: text,
    };
    return okJSON(out);
  } catch (e) {
    return serverError((e as Error).message);
  }
}

async function handleAnthropicMessages(req: Request) {
  let body: any;
  try {
    body = await req.json();
  } catch {
    return badRequest("Invalid JSON");
  }

  let norm: NormalizedInput;
  try {
    norm = tryFromAnthropic(body) ?? normalize(body);
  } catch (e) {
    return badRequest((e as Error).message);
  }

  try {
    const { text, finish } = await aggregateFromUpstream(norm);
    return okJSON(buildAnthropicMessage(norm.model, text, finish));
  } catch (e) {
    return serverError((e as Error).message);
  }
}

function handleModels() {
  return okJSON(modelsPayload());
}

function handleOptions() {
  return new Response(null, { status: 204, headers: CORS_HEADERS });
}

// ---- Server ----

const PORT = Number(Deno.env.get("PORT") ?? "8787");
console.log(`Server listening on http://localhost:${PORT}`);

Deno.serve({ port: PORT }, async (req) => {
  const url = new URL(req.url);

  if (req.method === "OPTIONS") return handleOptions();

  if (req.method === "GET" && url.pathname === "/v1/models") {
    return handleModels();
  }

  if (req.method === "POST" && url.pathname === "/v1/chat/completions") {
    return await handleChatCompletions(req);
  }

  if (req.method === "POST" && url.pathname === "/v1/responses") {
    return await handleResponses(req);
  }

  if (req.method === "POST" && url.pathname === "/v1/messages") {
    return await handleAnthropicMessages(req);
  }

  return new Response(JSON.stringify({ error: "Not Found" }), { status: 404, headers: jsonHeaders() });
});