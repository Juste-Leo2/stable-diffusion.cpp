export interface Resolution {
  id: string;
  label: string;
  aspect: string;
  width: number;
  height: number;
  tier: "1024" | "512";
}

export const RESOLUTIONS: Resolution[] = [
  { id: "2x1-1024", label: "2:1 — 1408 × 704", aspect: "2:1", width: 1408, height: 704, tier: "1024" },
  { id: "3x2-1024", label: "3:2 — 1248 × 832", aspect: "3:2", width: 1248, height: 832, tier: "1024" },
  { id: "1x1-1024", label: "1:1 — 1024 × 1024", aspect: "1:1", width: 1024, height: 1024, tier: "1024" },
  { id: "2x3-1024", label: "2:3 — 832 × 1248", aspect: "2:3", width: 832, height: 1248, tier: "1024" },
  { id: "1x2-1024", label: "1:2 — 704 × 1408", aspect: "1:2", width: 704, height: 1408, tier: "1024" },
  { id: "2x1-512", label: "2:1 — 704 × 352", aspect: "2:1", width: 704, height: 352, tier: "512" },
  { id: "3x2-512", label: "3:2 — 576 × 384", aspect: "3:2", width: 576, height: 384, tier: "512" },
  { id: "1x1-512", label: "1:1 — 512 × 512", aspect: "1:1", width: 512, height: 512, tier: "512" },
  { id: "2x3-512", label: "2:3 — 384 × 576", aspect: "2:3", width: 384, height: 576, tier: "512" },
  { id: "1x2-512", label: "1:2 — 352 × 704", aspect: "1:2", width: 352, height: 704, tier: "512" },
];

export const DEFAULT_RESOLUTION_ID = "1x1-512";

export const STORAGE_KEY_RESOLUTION = "bonsai:resolution-id";

export function resolutionById(id: string): Resolution {
  return RESOLUTIONS.find((r) => r.id === id) ?? RESOLUTIONS.find((r) => r.id === DEFAULT_RESOLUTION_ID)!;
}
