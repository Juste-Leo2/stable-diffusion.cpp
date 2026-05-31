import { MODERATION_REJECT_MESSAGE, validatePrompt } from "@/lib/prompt-moderator";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:8000";

export async function POST(request: Request) {
  const payload = await request.text();

  let parsed: { prompt?: unknown };
  try {
    parsed = JSON.parse(payload);
  } catch {
    return Response.json({ detail: "Invalid JSON body." }, { status: 400 });
  }
  if (typeof parsed.prompt === "string") {
    const verdict = validatePrompt(parsed.prompt);
    if (!verdict.ok && verdict.reason === "moderation") {
      return Response.json({ detail: MODERATION_REJECT_MESSAGE }, { status: 400 });
    }
  }

  try {
    const upstream = await fetch(`${BACKEND_URL}/generate/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
      cache: "no-store",
    });
    const contentType = upstream.headers.get("content-type") ?? "application/json";
    const body = await upstream.arrayBuffer();
    return new Response(body, {
      status: upstream.status,
      headers: { "Content-Type": contentType },
    });
  } catch {
    return Response.json(
      { detail: `Could not reach backend at ${BACKEND_URL}.` },
      { status: 502 },
    );
  }
}
