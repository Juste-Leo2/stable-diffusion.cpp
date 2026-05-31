// PNG iTXt-chunk injection. Embed keyword=value pairs with UTF-8 text after
// IHDR. iTXt (vs tEXt) is the spec-compliant choice for non-Latin-1 text — our
// prompts routinely contain Unicode (em-dashes, smart quotes, emoji, CJK).
// Spec: https://www.w3.org/TR/PNG/#11iTXt

const PNG_SIGNATURE = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

const CRC32_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) === 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    c = (CRC32_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8)) >>> 0;
  }
  return (c ^ 0xffffffff) >>> 0;
}

function readUint32BE(bytes: Uint8Array, offset: number): number {
  return (
    ((bytes[offset] << 24) >>> 0) +
    ((bytes[offset + 1] << 16) >>> 0) +
    ((bytes[offset + 2] << 8) >>> 0) +
    bytes[offset + 3]
  );
}

function makeITextChunk(keyword: string, text: string): Uint8Array {
  // iTXt payload (PNG spec §11.3.4.5):
  //   keyword (1-79 Latin-1 bytes) | 0x00 | comp_flag(1) | comp_method(1)
  //   | language_tag(Latin-1, null-terminated, may be empty)
  //   | translated_keyword(UTF-8, null-terminated, may be empty)
  //   | text(UTF-8, runs to end of chunk — no terminator)
  // Our keywords are ASCII; text is full UTF-8.
  const enc = new TextEncoder();
  const kw = enc.encode(keyword);
  if (kw.length < 1 || kw.length > 79) {
    throw new Error(`iTXt keyword must be 1-79 bytes (got ${kw.length})`);
  }
  const txt = enc.encode(text);
  // 5 trailing zero bytes: keyword-terminator, comp_flag=0, comp_method=0,
  // empty-language-terminator, empty-translated-keyword-terminator.
  const data = new Uint8Array(kw.length + 5 + txt.length);
  data.set(kw, 0);
  // data[kw.length .. kw.length+4] all zero by allocation default
  data.set(txt, kw.length + 5);

  const typeBytes = new Uint8Array([0x69, 0x54, 0x58, 0x74]); // "iTXt"
  const crcInput = new Uint8Array(typeBytes.length + data.length);
  crcInput.set(typeBytes, 0);
  crcInput.set(data, typeBytes.length);
  const crc = crc32(crcInput);

  const chunk = new Uint8Array(4 + 4 + data.length + 4);
  const dv = new DataView(chunk.buffer);
  dv.setUint32(0, data.length, false);
  chunk.set(typeBytes, 4);
  chunk.set(data, 8);
  dv.setUint32(8 + data.length, crc, false);
  return chunk;
}

export function addITextChunks(pngBytes: Uint8Array, kv: Record<string, string>): Uint8Array {
  for (let i = 0; i < 8; i++) {
    if (pngBytes[i] !== PNG_SIGNATURE[i]) {
      throw new Error("Not a PNG (signature mismatch)");
    }
  }
  // First chunk after the 8-byte signature must be IHDR. Skip it so the
  // injected iTXt chunks land between IHDR and IDAT (PNG spec section 5.3).
  const ihdrLength = readUint32BE(pngBytes, 8);
  const ihdrEnd = 8 + 4 + 4 + ihdrLength + 4; // sig + len + type + data + crc

  const chunks = Object.entries(kv)
    .filter(([, v]) => v.length > 0)
    .map(([k, v]) => makeITextChunk(k, v));
  const totalNew = chunks.reduce((a, c) => a + c.length, 0);

  const out = new Uint8Array(pngBytes.length + totalNew);
  out.set(pngBytes.subarray(0, ihdrEnd), 0);
  let off = ihdrEnd;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  out.set(pngBytes.subarray(ihdrEnd), off);
  return out;
}

export async function injectMetadata(blob: Blob, kv: Record<string, string>): Promise<Blob> {
  const buf = new Uint8Array(await blob.arrayBuffer());
  const enriched = addITextChunks(buf, kv);
  return new Blob([new Uint8Array(enriched)], { type: "image/png" });
}

interface MetadataSource {
  prompt: string;
  params: { seed: number; steps: number; backend: string; resolutionId: string };
  timestamp: number;
}

export function buildMetadata(source: MetadataSource, resolutionLabel: string): Record<string, string> {
  return {
    prompt: source.prompt,
    seed: String(source.params.seed),
    steps: String(source.params.steps),
    backend: source.params.backend,
    resolution: resolutionLabel,
    timestamp: new Date(source.timestamp).toISOString(),
    model: "Bonsai Image",
  };
}

export async function downloadWithMetadata(
  blob: Blob,
  filename: string,
  kv: Record<string, string>,
): Promise<void> {
  const enriched = await injectMetadata(blob, kv);
  const url = URL.createObjectURL(enriched);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  // Revoke after a tick so the browser has time to start the download.
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}
