export type BackendKind = "mlx" | "gemlite";
export type ModelFamily =
  | "bonsai-binary"
  | "bonsai-ternary";

export interface Backend {
  value: string;
  label: string;
  hint: string;
  kind: BackendKind;
  modelFamily: ModelFamily;
}

export const BACKENDS: Backend[] = [
  {
    value: "bonsai-ternary-mlx",
    label: "Bonsai · Ternary",
    hint: "Compact MLX weights for local Mac rendering.",
    kind: "mlx",
    modelFamily: "bonsai-ternary",
  },
  {
    value: "bonsai-binary-mlx",
    label: "Bonsai · Binary",
    hint: "1-bit Klein MLX weights for local Mac rendering.",
    kind: "mlx",
    modelFamily: "bonsai-binary",
  },
  {
    value: "bonsai-ternary-gemlite",
    label: "Bonsai · Ternary (GPU)",
    hint: "2-bit Klein via gemlite/HQQ on a remote H100.",
    kind: "gemlite",
    modelFamily: "bonsai-ternary",
  },
  {
    value: "bonsai-binary-gemlite",
    label: "Bonsai · Binary (GPU)",
    hint: "1-bit Klein via gemlite/HQQ on a remote H100.",
    kind: "gemlite",
    modelFamily: "bonsai-binary",
  },
];

export const DEFAULT_BACKEND: string = BACKENDS[0].value;

export const BACKEND_KINDS: { value: BackendKind; label: string }[] = [
  { value: "mlx", label: "MLX · Mac" },
  { value: "gemlite", label: "GPU · Remote" },
];

export const MODEL_FAMILIES: { value: ModelFamily; label: string }[] = [
  { value: "bonsai-binary", label: "Bonsai · Binary" },
  { value: "bonsai-ternary", label: "Bonsai · Ternary" },
];

export function backendFor(
  kind: BackendKind,
  modelFamily: ModelFamily,
): Backend | undefined {
  return BACKENDS.find((b) => b.kind === kind && b.modelFamily === modelFamily);
}

export function migrateBackendId(value: string): string {
  if (BACKENDS.some((b) => b.value === value)) return value;
  return DEFAULT_BACKEND;
}
