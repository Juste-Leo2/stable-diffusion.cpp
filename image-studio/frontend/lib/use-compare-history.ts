"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export interface PendingSlot {
  backend: string;
  status: "pending";
}

export interface ReadySlot {
  backend: string;
  status: "ready";
  imageUrl: string;
  wallSeconds: number;
  swapSeconds: number;
}

export interface ErrorSlot {
  backend: string;
  status: "error";
  error: string;
}

export type CompareSlot = PendingSlot | ReadySlot | ErrorSlot;

export interface CompareParams {
  seed: number;
  steps: number;
  guidance: number;
  resolutionId: string;
}

export interface CompareEntry {
  id: string;
  prompt: string;
  presetId: string | null;
  params: CompareParams;
  results: CompareSlot[];
  timestamp: number;
}

export const MAX_COMPARE_HISTORY = 10;

function newId() {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function readyUrls(entry: CompareEntry): string[] {
  return entry.results.flatMap((s) => (s.status === "ready" ? [s.imageUrl] : []));
}

export function useCompareHistory() {
  const [entries, setEntries] = useState<CompareEntry[]>([]);
  const ownedRef = useRef<Set<string>>(new Set());

  const push = useCallback(
    (entry: Omit<CompareEntry, "id" | "timestamp">): CompareEntry => {
      const full: CompareEntry = { ...entry, id: newId(), timestamp: Date.now() };
      for (const url of readyUrls(full)) ownedRef.current.add(url);
      setEntries((prev) => {
        const next = [full, ...prev];
        while (next.length > MAX_COMPARE_HISTORY) {
          const evicted = next.pop()!;
          for (const url of readyUrls(evicted)) {
            if (ownedRef.current.delete(url)) URL.revokeObjectURL(url);
          }
        }
        return next;
      });
      return full;
    },
    [],
  );

  const updateEntry = useCallback(
    (id: string, patch: (prev: CompareEntry) => CompareEntry) => {
      setEntries((prev) =>
        prev.map((e) => {
          if (e.id !== id) return e;
          const next = patch(e);
          for (const url of readyUrls(next)) ownedRef.current.add(url);
          return next;
        }),
      );
    },
    [],
  );

  const clear = useCallback(() => {
    setEntries((prev) => {
      for (const e of prev) {
        for (const url of readyUrls(e)) {
          if (ownedRef.current.delete(url)) URL.revokeObjectURL(url);
        }
      }
      return [];
    });
  }, []);

  useEffect(() => {
    const owned = ownedRef.current;
    return () => {
      for (const url of owned) URL.revokeObjectURL(url);
      owned.clear();
    };
  }, []);

  return { entries, push, updateEntry, clear } as const;
}
