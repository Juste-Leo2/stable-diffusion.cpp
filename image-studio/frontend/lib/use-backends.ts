"use client";

import { useEffect, useState } from "react";
import {
  BACKENDS,
  MODEL_FAMILIES,
  backendFor,
  type Backend,
  type BackendKind,
  type ModelFamily,
} from "@/lib/backends";

interface BackendsAvailability {
  kind: BackendKind;
  supported_families: ModelFamily[];
  default_family: ModelFamily;
  healthy: boolean;
  reason: string | null;
}

interface UseBackendsResult {
  // New API:
  kind: BackendKind;
  supportedFamilies: ModelFamily[];
  defaultFamily: ModelFamily;
  // Derived for legacy consumers (compare-client + studio-client transition):
  backends: Backend[];
  defaultBackend: string;
  // Health:
  healthy: boolean;
  reason: string | null;
  loaded: boolean;
}

// MLX-only fallback so the picker stays usable if /api/backends fails on Mac.
const FALLBACK: BackendsAvailability = {
  kind: "mlx",
  supported_families: MODEL_FAMILIES.map((f) => f.value),
  default_family: "bonsai-ternary",
  healthy: false,
  reason: "discovery_failed",
};

let cached: BackendsAvailability | null = null;

async function fetchBackends(): Promise<BackendsAvailability> {
  const response = await fetch("/api/backends", { cache: "no-store" });
  if (!response.ok) throw new Error(`/api/backends: ${response.status}`);
  return (await response.json()) as BackendsAvailability;
}

export function useBackends(): UseBackendsResult {
  const [availability, setAvailability] = useState<BackendsAvailability | null>(cached);
  const [loaded, setLoaded] = useState<boolean>(cached !== null);

  useEffect(() => {
    let cancelled = false;
    fetchBackends()
      .then((value) => {
        if (cancelled) return;
        cached = value;
        setAvailability(value);
        setLoaded(true);
      })
      .catch(() => {
        if (cancelled) return;
        setAvailability(FALLBACK);
        setLoaded(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const resolved = availability ?? FALLBACK;
  const kind: BackendKind = resolved.kind ?? FALLBACK.kind;
  // Render in MODEL_FAMILIES order, filtered to families the server actually
  // supports. The server already returns them in that canonical order, but
  // this re-sorts defensively in case it ever lists them differently.
  const supportedFamilies: ModelFamily[] = MODEL_FAMILIES
    .map((f) => f.value)
    .filter((f) => resolved.supported_families?.includes(f) ?? true);
  const defaultFamily = supportedFamilies.includes(resolved.default_family)
    ? resolved.default_family
    : (supportedFamilies[0] ?? MODEL_FAMILIES[0].value);

  const backends: Backend[] = supportedFamilies
    .map((family) => backendFor(kind, family))
    .filter((b): b is Backend => b !== undefined);
  const defaultBackendEntry = backendFor(kind, defaultFamily);
  const defaultBackend = defaultBackendEntry?.value ?? backends[0]?.value ?? BACKENDS[0].value;

  return {
    kind,
    supportedFamilies,
    defaultFamily,
    backends,
    defaultBackend,
    healthy: resolved.healthy,
    reason: resolved.reason,
    loaded,
  };
}
