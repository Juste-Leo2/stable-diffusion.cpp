import { MODERATION_REJECT_MESSAGE, validatePrompt } from "@/lib/prompt-moderator";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:8000";

export async function POST(request: Request) {
  const payload = await request.text();

  // Server-side moderation gate: client may be tampered with, so this is the
  // real enforcement. The client also runs validatePrompt for snappier UX.
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
    const upstream = await fetch(`${BACKEND_URL}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
      cache: "no-store",
    });
    const contentType = upstream.headers.get("content-type") ?? "application/octet-stream";
    const body = await upstream.arrayBuffer();
    const headers: Record<string, string> = { "Content-Type": contentType };
    const peak = upstream.headers.get("x-peak-memory-mb");
    const wall = upstream.headers.get("x-wall-seconds");
    if (peak) headers["X-Peak-Memory-MB"] = peak;
    if (wall) headers["X-Wall-Seconds"] = wall;
    return new Response(body, { status: upstream.status, headers });
  } catch {
    return Response.json(
      { detail: `Could not reach backend at ${BACKEND_URL}.` },
      { status: 502 },
    );
  }
}
