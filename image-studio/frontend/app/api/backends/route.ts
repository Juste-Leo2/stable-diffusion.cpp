const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:8000";

export async function GET(request: Request) {
  const incoming = new URL(request.url);
  const upstreamURL = new URL(`${BACKEND_URL}/backends`);
  const force = incoming.searchParams.get("force_disable");
  if (force === "1" || force === "true") {
    upstreamURL.searchParams.set("force_disable", "1");
  }

  try {
    const upstream = await fetch(upstreamURL.toString(), {
      method: "GET",
      cache: "no-store",
    });
    const body = await upstream.text();
    return new Response(body, {
      status: upstream.status,
      headers: { "Content-Type": upstream.headers.get("content-type") ?? "application/json" },
    });
  } catch {
    return Response.json(
      { detail: `Could not reach backend at ${BACKEND_URL}.` },
      { status: 502 },
    );
  }
}
