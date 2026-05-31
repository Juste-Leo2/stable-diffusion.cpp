"use client";

import { useCallback, useSyncExternalStore } from "react";
import { idbClear, idbDelete, idbGetAll, idbPut } from "./idb-history";

export interface HistoryParams {
  seed: number;
  steps: number;
  backend: string;
  resolutionId: string;
}

export interface HistoryEntry {
  id: string;
  prompt: string;
  params: HistoryParams;
  imageBlob: Blob;
  imageUrl: string;
  timestamp: number;
}

export const MAX_HISTORY = 250;

// ── Module-level store ────────────────────────────────────────────────────────
// Lives outside React so history survives navigation (component unmounts). On
// first subscribe we async-hydrate from IndexedDB so reload also preserves it.

let _entries: HistoryEntry[] = [];
const _listeners = new Set<() => void>();
const _owned = new Set<string>();
let _hydrateStarted = false;

function notify() {
  for (const l of _listeners) l();
}

function getSnapshot() {
  return _entries;
}

function subscribe(listener: () => void) {
  if (!_hydrateStarted) {
    _hydrateStarted = true;
    void hydrate();
  }
  _listeners.add(listener);
  return () => {
    _listeners.delete(listener);
  };
}

async function hydrate() {
  try {
    const stored = await idbGetAll();
    if (stored.length === 0) return;
    stored.sort((a, b) => b.timestamp - a.timestamp);
    const inMemoryIds = new Set(_entries.map((e) => e.id));
    const hydrated: HistoryEntry[] = stored.slice(0, MAX_HISTORY).flatMap((s) => {
      // Skip anything already present (rare race: a render finished before hydrate)
      if (inMemoryIds.has(s.id)) return [];
      const url = URL.createObjectURL(s.imageBlob);
      _owned.add(url);
      return [{
        id: s.id,
        prompt: s.prompt,
        params: s.params,
        imageBlob: s.imageBlob,
        imageUrl: url,
        timestamp: s.timestamp,
      }];
    });
    if (hydrated.length === 0) {
      notify();
      return;
    }
    const merged = [..._entries, ...hydrated];
    merged.sort((a, b) => b.timestamp - a.timestamp);
    // Revoke + clean ownership for entries that fall past the MAX_HISTORY tail —
    // otherwise objectURLs created for those hydrated blobs leak.
    const drop = merged.slice(MAX_HISTORY);
    for (const e of drop) {
      if (_owned.delete(e.imageUrl)) URL.revokeObjectURL(e.imageUrl);
    }
    _entries = merged.slice(0, MAX_HISTORY);
    notify();
  } catch (err) {
    console.warn("history hydrate failed", err);
  }
}

interface PushInput {
  prompt: string;
  params: HistoryParams;
  imageBlob: Blob;
}

function pushEntry(input: PushInput): HistoryEntry {
  const id =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const timestamp = Date.now();
  const imageUrl = URL.createObjectURL(input.imageBlob);
  _owned.add(imageUrl);
  const full: HistoryEntry = {
    id,
    prompt: input.prompt,
    params: input.params,
    imageBlob: input.imageBlob,
    imageUrl,
    timestamp,
  };
  _entries = [full, ..._entries];
  void idbPut({
    id,
    prompt: input.prompt,
    params: input.params,
    imageBlob: input.imageBlob,
    timestamp,
  }).catch((e) => console.warn("idbPut failed", e));
  while (_entries.length > MAX_HISTORY) {
    const evicted = _entries.pop()!;
    if (_owned.delete(evicted.imageUrl)) URL.revokeObjectURL(evicted.imageUrl);
    void idbDelete(evicted.id).catch((e) => console.warn("idbDelete (evict) failed", e));
  }
  notify();
  return full;
}

function clearEntries() {
  for (const e of _entries) {
    if (_owned.delete(e.imageUrl)) URL.revokeObjectURL(e.imageUrl);
  }
  _entries = [];
  void idbClear().catch((e) => console.warn("idbClear failed", e));
  notify();
}

function removeEntry(id: string) {
  const target = _entries.find((e) => e.id === id);
  if (!target) return;
  if (_owned.delete(target.imageUrl)) URL.revokeObjectURL(target.imageUrl);
  _entries = _entries.filter((e) => e.id !== id);
  void idbDelete(id).catch((e) => console.warn("idbDelete failed", e));
  notify();
}

const EMPTY: HistoryEntry[] = [];
function getServerSnapshot() {
  return EMPTY;
}

// ── Hook ──────────────────────────────────────────────────────────────────────

export function useHistory() {
  const entries = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  const push = useCallback((input: PushInput) => pushEntry(input), []);
  const clear = useCallback(() => clearEntries(), []);
  const remove = useCallback((id: string) => removeEntry(id), []);

  return { entries, push, clear, remove } as const;
}
