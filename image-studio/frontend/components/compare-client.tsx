"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, ChevronDown, Dice5, Download, LoaderCircle, Save, Sparkles } from "lucide-react";
import {
  BACKENDS,
  MODEL_FAMILIES,
  backendFor,
  type ModelFamily,
} from "@/lib/backends";
import { MODERATION_REJECT_MESSAGE, validatePrompt } from "@/lib/prompt-moderator";
import { useBackends } from "@/lib/use-backends";
import { COMPARE_PRESETS, DEFAULT_PRESET_ID } from "@/lib/compare-presets";
import { DEFAULT_RESOLUTION_ID, RESOLUTIONS, resolutionById } from "@/lib/resolutions";
import {
  type CompareEntry,
  type CompareSlot,
  type ErrorSlot,
  type PendingSlot,
  type ReadySlot,
  useCompareHistory,
} from "@/lib/use-compare-history";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

const COMPARE_PATH = "/api/generate/compare";

interface CompareApiResult {
  backend: string;
  png_b64: string;
  wall_seconds: number;
  swap_seconds: number;
}

interface CompareApiResponse {
  results: CompareApiResult[];
}

async function parseError(response: Response) {
  const text = await response.text().catch(() => "");
  try {
    const parsed = JSON.parse(text) as { detail?: string };
    if (typeof parsed.detail === "string" && parsed.detail.length > 0) return parsed.detail;
  } catch {
    // fall through
  }
  return text || `Request failed with status ${response.status}.`;
}

function b64ToBlob(b64: string) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: "image/png" });
}

function backendLabel(value: string) {
  return BACKENDS.find((b) => b.value === value)?.label ?? value;
}

function formatSeconds(s: number) {
  return `${s.toFixed(2)}s`;
}

function readToken(name: string, fallback: string): string {
  // Canvas can't consume CSS variables directly — read the live value off :root
  // so the stitched export picks up whichever theme is active.
  if (typeof window === "undefined") return fallback;
  const resolved = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return resolved || fallback;
}

async function stitchResults(entry: CompareEntry): Promise<Blob> {
  // Compose ready slots horizontally with a small gutter + caption strip at the
  // bottom. Pending/error slots are skipped; caller should gate on ready count.
  const readySlots = entry.results.filter((s): s is ReadySlot => s.status === "ready");
  if (readySlots.length === 0) throw new Error("No completed images to stitch yet.");

  const gutter = 16;
  const captionHeight = 56;

  const images = await Promise.all(
    readySlots.map(
      (r) =>
        new Promise<HTMLImageElement>((resolve, reject) => {
          const img = new Image();
          img.onload = () => resolve(img);
          img.onerror = () => reject(new Error(`Failed to load ${r.backend}`));
          img.src = r.imageUrl;
        }),
    ),
  );

  const frameWidth = Math.max(...images.map((i) => i.naturalWidth));
  const frameHeight = Math.max(...images.map((i) => i.naturalHeight));
  const n = images.length;
  const canvas = document.createElement("canvas");
  canvas.width = frameWidth * n + gutter * (n + 1);
  canvas.height = frameHeight + captionHeight + gutter * 2;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2D canvas context unavailable");

  const bgColor = readToken("--background", "#101014");
  const mutedStrong = readToken("--muted-strong", "#d4d4d8");
  const muted = readToken("--muted", "#a1a1aa");

  ctx.fillStyle = bgColor;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  images.forEach((img, i) => {
    const x = gutter + i * (frameWidth + gutter);
    ctx.drawImage(img, x, gutter, frameWidth, frameHeight);
    const result = readySlots[i];
    ctx.fillStyle = mutedStrong;
    ctx.font = "14px ui-sans-serif, system-ui, sans-serif";
    ctx.textBaseline = "top";
    ctx.fillText(
      backendLabel(result.backend),
      x,
      gutter + frameHeight + 8,
    );
    ctx.fillStyle = muted;
    ctx.font = "12px ui-monospace, monospace";
    ctx.fillText(
      `wall ${formatSeconds(result.wallSeconds)} · swap ${formatSeconds(result.swapSeconds)}`,
      x,
      gutter + frameHeight + 30,
    );
  });

  return await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) reject(new Error("Canvas toBlob returned null"));
      else resolve(blob);
    }, "image/png");
  });
}

export function CompareClient() {
  const [presetId, setPresetId] = useState<string>(DEFAULT_PRESET_ID);
  const [prompt, setPrompt] = useState<string>(
    COMPARE_PRESETS.find((p) => p.id === DEFAULT_PRESET_ID)?.prompt ?? "",
  );
  const [seed, setSeed] = useState(42);
  const [steps, setSteps] = useState(4);
  const [guidance, setGuidance] = useState(1.0);
  const [resolutionId, setResolutionId] = useState<string>(DEFAULT_RESOLUTION_ID);
  // The relay reports one kind per process; compare picks across families
  // within that kind. Switching kinds requires restarting the relay.
  const { kind, supportedFamilies, defaultFamily } = useBackends();
  const [selectedFamilies, setSelectedFamilies] = useState<ModelFamily[]>([defaultFamily]);
  const familyOptions = useMemo(
    () => MODEL_FAMILIES.filter((f) => supportedFamilies.includes(f.value)),
    [supportedFamilies],
  );
  // Sync to the server-resolved family list after /api/backends lands and
  // whenever the user's choices drift outside the supported set.
  useEffect(() => {
    if (!supportedFamilies.length) return;
    setSelectedFamilies((prev) => {
      const filtered = prev.filter((f) => supportedFamilies.includes(f));
      if (filtered.length === 0) return [defaultFamily];
      return filtered;
    });
  }, [supportedFamilies, defaultFamily]);
  const effectiveSelectedBackends = useMemo(() => {
    // Preserve canonical family order for deterministic slot layout.
    return MODEL_FAMILIES
      .filter((f) => selectedFamilies.includes(f.value))
      .map((f) => backendFor(kind, f.value)?.value)
      .filter((v): v is string => v !== undefined);
  }, [selectedFamilies, kind]);

  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeEntryId, setActiveEntryId] = useState<string | null>(null);

  const { entries, push, updateEntry, clear } = useCompareHistory();
  const promptRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = promptRef.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${el.scrollHeight}px`;
  }, [prompt]);

  const resolution = useMemo(() => resolutionById(resolutionId), [resolutionId]);
  const sizeLabel = `${resolution.width} × ${resolution.height}`;
  const activeEntry = useMemo(
    () => entries.find((e) => e.id === activeEntryId) ?? entries[0] ?? null,
    [entries, activeEntryId],
  );

  const selectPreset = useCallback((id: string) => {
    setPresetId(id);
    const preset = COMPARE_PRESETS.find((p) => p.id === id);
    if (preset) setPrompt(preset.prompt);
  }, []);

  const toggleFamily = useCallback((value: ModelFamily) => {
    setSelectedFamilies((prev) => {
      if (prev.includes(value)) {
        if (prev.length === 1) return prev;
        return prev.filter((v) => v !== value);
      }
      // Preserve canonical order from MODEL_FAMILIES.
      return MODEL_FAMILIES
        .filter((f) => prev.includes(f.value) || f.value === value)
        .map((f) => f.value);
    });
  }, []);

  const handleRun = useCallback(async () => {
    if (isRunning || prompt.trim().length === 0) return;
    const verdict = validatePrompt(prompt);
    if (!verdict.ok && verdict.reason === "moderation") {
      setError(MODERATION_REJECT_MESSAGE);
      return;
    }
    setIsRunning(true);
    setError(null);

    // effectiveSelectedBackends is already in canonical family order.
    const backends = effectiveSelectedBackends;

    const entry = push({
      prompt,
      presetId: presetId && COMPARE_PRESETS.some((p) => p.id === presetId) ? presetId : null,
      params: { seed, steps, guidance, resolutionId },
      results: backends.map<PendingSlot>((backend) => ({ backend, status: "pending" })),
    });
    setActiveEntryId(entry.id);

    for (const backend of backends) {
      const response = await fetch(COMPARE_PATH, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          seed,
          steps,
          guidance,
          height: resolution.height,
          width: resolution.width,
          backends: [backend],
        }),
      }).catch(() => null);

      const patchSlot = (slot: CompareSlot): CompareSlot => {
        if (slot.backend !== backend) return slot;
        if (!response) {
          return { backend, status: "error", error: "Could not reach backend." };
        }
        return slot;
      };

      if (!response) {
        updateEntry(entry.id, (prev) => ({ ...prev, results: prev.results.map(patchSlot) }));
        continue;
      }
      if (!response.ok) {
        const message = await parseError(response);
        updateEntry(entry.id, (prev) => ({
          ...prev,
          results: prev.results.map<CompareSlot>((s) =>
            s.backend === backend ? { backend, status: "error", error: message } : s,
          ),
        }));
        continue;
      }

      const json = (await response.json()) as CompareApiResponse;
      const r = json.results.find((x) => x.backend === backend) ?? json.results[0];
      if (!r) {
        updateEntry(entry.id, (prev) => ({
          ...prev,
          results: prev.results.map<CompareSlot>((s) =>
            s.backend === backend ? { backend, status: "error", error: "Empty response." } : s,
          ),
        }));
        continue;
      }
      const imageUrl = URL.createObjectURL(b64ToBlob(r.png_b64));
      updateEntry(entry.id, (prev) => ({
        ...prev,
        results: prev.results.map<CompareSlot>((s) =>
          s.backend === backend
            ? {
                backend,
                status: "ready",
                imageUrl,
                wallSeconds: r.wall_seconds,
                swapSeconds: r.swap_seconds,
              }
            : s,
        ),
      }));
    }

    setIsRunning(false);
  }, [
    guidance,
    isRunning,
    presetId,
    prompt,
    push,
    resolution.height,
    resolution.width,
    resolutionId,
    seed,
    effectiveSelectedBackends,
    steps,
    updateEntry,
  ]);

  const readyCount = activeEntry?.results.filter((s) => s.status === "ready").length ?? 0;

  const handleSaveAll = useCallback(async () => {
    if (!activeEntry || readyCount === 0) return;
    try {
      const blob = await stitchResults(activeEntry);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `bonsai-compare-${new Date(activeEntry.timestamp)
        .toISOString()
        .replace(/[:.]/g, "-")}.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      // Give the download a tick before revoking.
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to stitch results.");
    }
  }, [activeEntry, readyCount]);

  return (
    <div className="flex flex-col gap-5">
      <section className="rounded-3xl bg-surface p-5 backdrop-blur-xl sm:p-6">
        <div className="mb-3 flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm font-medium text-muted">
            <Sparkles className="size-4 text-accent" />
            Compare prompt across backends
          </div>
          <span className="rounded-full border border-border-strong bg-surface-raised px-2.5 py-1 font-mono text-[10px] uppercase tracking-[0.22em] text-muted">
            internal · power user
          </span>
        </div>

        <div className="mb-3 space-y-2">
          <p className="text-xs uppercase tracking-[0.22em] text-muted">Preset</p>
          <div className="flex flex-wrap gap-2">
            {COMPARE_PRESETS.map((preset) => {
              const selected = preset.id === presetId;
              return (
                <button
                  key={preset.id}
                  type="button"
                  onClick={() => selectPreset(preset.id)}
                  className={cn(
                    "rounded-full border px-3 py-1.5 text-xs font-medium transition",
                    selected
                      ? "border-accent bg-accent-soft text-accent-strong"
                      : "border-border-strong bg-surface-raised text-muted-strong hover:border-accent/50",
                  )}
                >
                  {preset.label}
                </button>
              );
            })}
          </div>
        </div>

        <Textarea
          ref={promptRef}
          className="min-h-[120px]"
          placeholder="Pick a preset or type a custom prompt…"
          value={prompt}
          onChange={(e) => {
            setPrompt(e.target.value);
            setPresetId("");
          }}
        />

        <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <label className="space-y-1.5 text-xs">
            <span className="flex items-center justify-between text-muted">
              <span>Seed</span>
              <button
                type="button"
                onClick={() => setSeed(Math.floor(Math.random() * 2 ** 31))}
                className="flex items-center gap-1 rounded-full border border-border-strong bg-surface-raised px-2 py-0.5 text-[10px] font-medium text-muted-strong transition hover:border-accent/60 hover:text-accent-strong"
              >
                <Dice5 className="size-3" />
                Random
              </button>
            </span>
            <Input
              type="number"
              min={0}
              step={1}
              value={seed}
              onChange={(e) => {
                const parsed = Number.parseInt(e.target.value, 10);
                if (Number.isFinite(parsed)) setSeed(Math.max(0, parsed));
              }}
            />
          </label>

          <label className="space-y-1.5 text-xs">
            <span className="text-muted">Steps</span>
            <Input
              type="number"
              min={1}
              step={1}
              value={steps}
              onChange={(e) => {
                const parsed = Number.parseInt(e.target.value, 10);
                if (Number.isFinite(parsed)) setSteps(Math.max(1, parsed));
              }}
            />
          </label>

          <label className="space-y-1.5 text-xs">
            <span className="text-muted">Guidance</span>
            <Input
              type="number"
              min={0}
              step={0.1}
              value={guidance}
              onChange={(e) => {
                const parsed = Number.parseFloat(e.target.value);
                if (Number.isFinite(parsed)) setGuidance(Math.max(0, parsed));
              }}
            />
          </label>

          <label className="space-y-1.5 text-xs">
            <span className="text-muted">Resolution</span>
            <div className="relative">
              <select
                className="h-11 w-full appearance-none rounded-2xl border border-border-strong bg-surface-raised pl-4 pr-10 text-sm text-foreground outline-none transition focus:border-accent focus:ring-2 focus:ring-accent-ring"
                value={resolutionId}
                onChange={(e) => setResolutionId(e.target.value)}
              >
                {RESOLUTIONS.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.label}
                  </option>
                ))}
              </select>
              <ChevronDown
                aria-hidden
                className="pointer-events-none absolute right-3 top-1/2 size-4 -translate-y-1/2 text-muted"
              />
            </div>
          </label>
        </div>

        <div className="mt-4 space-y-2">
          <div className="flex items-center justify-between">
            <p className="text-xs uppercase tracking-[0.22em] text-muted">Models</p>
            <p className="font-mono text-[10px] text-muted">kind · {kind}</p>
          </div>
          <div className="flex flex-wrap gap-2">
            {familyOptions.map((f) => {
              const selected = selectedFamilies.includes(f.value);
              return (
                <button
                  key={f.value}
                  type="button"
                  onClick={() => toggleFamily(f.value)}
                  className={cn(
                    "rounded-full border px-3 py-1.5 text-xs font-medium transition",
                    selected
                      ? "border-accent bg-accent-soft text-accent-strong"
                      : "border-border-strong bg-surface-raised text-muted-strong hover:border-accent/50",
                  )}
                >
                  {f.label}
                </button>
              );
            })}
          </div>
        </div>

        <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-xs leading-5 text-muted">
            Runs {effectiveSelectedBackends.length} backend{effectiveSelectedBackends.length === 1 ? "" : "s"} at{" "}
            <span className="font-medium text-foreground">{sizeLabel}</span>, serialized.
          </p>
          <Button
            className="relative min-w-[220px]"
            disabled={isRunning || prompt.trim().length === 0 || effectiveSelectedBackends.length === 0}
            size="lg"
            type="button"
            onClick={handleRun}
          >
            {isRunning ? (
              <LoaderCircle className="size-5 animate-spin" />
            ) : (
              <Sparkles className="size-5" />
            )}
            <span>{isRunning ? "Running comparison" : "Run comparison"}</span>
          </Button>
        </div>

        {error ? (
          <p className="mt-3 rounded-2xl border border-danger/40 bg-danger-soft px-3 py-2 text-sm text-[#fca5a5]">
            {error}
          </p>
        ) : null}
      </section>

      <ComparePanel entry={activeEntry} readyCount={readyCount} onSaveAll={handleSaveAll} />

      <CompareHistoryStrip
        activeId={activeEntry?.id ?? null}
        entries={entries}
        onClear={() => {
          clear();
          setActiveEntryId(null);
        }}
        onSelect={(id) => setActiveEntryId(id)}
      />
    </div>
  );
}

function ComparePanel({
  entry,
  readyCount,
  onSaveAll,
}: {
  entry: CompareEntry | null;
  readyCount: number;
  onSaveAll: () => void;
}) {
  if (!entry) {
    return (
      <section className="rounded-3xl bg-surface p-8 text-center backdrop-blur-xl">
        <p className="text-sm text-muted">
          Run a comparison — results land here as a 3-up grid.
        </p>
      </section>
    );
  }

  return (
    <section className="rounded-3xl bg-surface p-5 backdrop-blur-xl sm:p-6">
      <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.22em] text-muted">Comparison</p>
          <p className="mt-1 line-clamp-2 max-w-[60ch] text-sm leading-6 text-foreground">
            {entry.prompt}
          </p>
        </div>
        <Button
          size="sm"
          variant="outline"
          type="button"
          onClick={onSaveAll}
          disabled={readyCount === 0}
        >
          <Save className="size-4" />
          Save all (stitched)
        </Button>
      </div>

      <div
        className="grid gap-4"
        style={{
          gridTemplateColumns: `repeat(${Math.max(entry.results.length, 1)}, minmax(0, 1fr))`,
        }}
      >
        {entry.results.map((slot) => (
          <CompareFrame key={slot.backend} slot={slot} />
        ))}
      </div>
    </section>
  );
}

function CompareFrame({ slot }: { slot: CompareSlot }) {
  if (slot.status === "pending") return <CompareFrameSkeleton backend={slot.backend} />;
  if (slot.status === "error") return <CompareFrameError slot={slot} />;
  return <CompareFrameReady slot={slot} />;
}

function CompareFrameReady({ slot }: { slot: ReadySlot }) {
  return (
    <figure className="space-y-2 rounded-2xl bg-surface-raised p-3 backdrop-blur-md">
      <div className="overflow-hidden rounded-xl bg-black/30">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img alt={slot.backend} className="h-auto w-full object-contain" src={slot.imageUrl} />
      </div>
      <figcaption className="space-y-0.5 px-1">
        <p className="text-xs font-medium text-foreground">{backendLabel(slot.backend)}</p>
        <p className="font-mono text-[10px] text-muted">
          wall {formatSeconds(slot.wallSeconds)} · swap {formatSeconds(slot.swapSeconds)}
        </p>
      </figcaption>
    </figure>
  );
}

function CompareFrameSkeleton({ backend }: { backend?: string }) {
  return (
    <div className="space-y-2 rounded-2xl bg-surface-raised p-3 backdrop-blur-md">
      <div className="relative aspect-square overflow-hidden rounded-xl bg-surface-strong">
        <div className="absolute inset-y-0 w-1/2 bg-[linear-gradient(90deg,transparent,var(--shimmer),transparent)] animate-[bonsai-shimmer_2.4s_linear_infinite]" />
        {backend ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="flex items-center gap-2 rounded-full border border-border-strong bg-surface px-3 py-1 font-mono text-[10px] uppercase tracking-[0.2em] text-muted backdrop-blur">
              <LoaderCircle className="size-3 animate-spin" />
              {backendLabel(backend)}
            </span>
          </div>
        ) : null}
      </div>
      <div className="space-y-1 px-1">
        {backend ? (
          <>
            <p className="text-xs font-medium text-foreground">{backendLabel(backend)}</p>
            <p className="font-mono text-[10px] text-muted">waiting…</p>
          </>
        ) : (
          <>
            <div className="h-3 w-24 rounded-full bg-white/8" />
            <div className="h-2.5 w-32 rounded-full bg-white/8" />
          </>
        )}
      </div>
    </div>
  );
}

function CompareFrameError({ slot }: { slot: ErrorSlot }) {
  return (
    <div className="space-y-2 rounded-2xl border border-danger/40 bg-danger-soft p-3">
      <div className="flex aspect-square items-center justify-center rounded-xl bg-black/20 p-4 text-center">
        <div className="space-y-2">
          <AlertTriangle aria-hidden className="mx-auto size-6 text-[#fca5a5]" />
          <p className="line-clamp-4 text-[11px] leading-4 text-[#fca5a5]">{slot.error}</p>
        </div>
      </div>
      <div className="space-y-0.5 px-1">
        <p className="text-xs font-medium text-foreground">{backendLabel(slot.backend)}</p>
        <p className="font-mono text-[10px] text-[#fca5a5]">failed</p>
      </div>
    </div>
  );
}

function CompareHistoryStrip({
  entries,
  activeId,
  onSelect,
  onClear,
}: {
  entries: CompareEntry[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onClear: () => void;
}) {
  if (entries.length === 0) return null;

  return (
    <section className="rounded-3xl bg-surface p-4 backdrop-blur-xl">
      <div className="mb-3 flex items-center justify-between">
        <p className="text-xs uppercase tracking-[0.22em] text-muted">
          Recent comparisons · {entries.length}/10
        </p>
        <button
          type="button"
          onClick={onClear}
          className="flex items-center gap-1.5 rounded-full border border-border-strong bg-surface-raised px-2.5 py-1 text-[11px] font-medium text-muted-strong transition hover:border-danger/50 hover:text-[#fca5a5]"
        >
          Clear
        </button>
      </div>
      <div className="flex gap-2 overflow-x-auto pb-2">
        {entries.map((entry) => {
          const isActive = entry.id === activeId;
          const thumbs = entry.results.filter((s): s is ReadySlot => s.status === "ready");
          return (
            <button
              key={entry.id}
              type="button"
              onClick={() => onSelect(entry.id)}
              className={cn(
                "group flex min-w-[220px] items-center gap-2 rounded-2xl border p-2 text-left transition",
                isActive
                  ? "border-accent bg-accent-soft"
                  : "border-border-strong bg-surface-raised hover:border-accent/50",
              )}
              title={entry.prompt}
            >
              <div className="flex shrink-0 gap-0.5">
                {thumbs.length > 0 ? (
                  thumbs.map((r) => (
                    /* eslint-disable-next-line @next/next/no-img-element */
                    <img
                      key={r.backend}
                      alt={r.backend}
                      src={r.imageUrl}
                      className="size-10 rounded object-cover"
                    />
                  ))
                ) : (
                  <div className="size-10 rounded bg-surface-strong" />
                )}
              </div>
              <div className="min-w-0 flex-1 space-y-0.5">
                <p className="line-clamp-2 text-[11px] leading-4 text-foreground">
                  {entry.prompt || <span className="italic text-muted">untitled</span>}
                </p>
                <p className="font-mono text-[10px] tracking-wide text-muted">
                  seed {entry.params.seed} · {entry.params.steps}s
                </p>
              </div>
              <Download
                aria-hidden
                className="size-3.5 text-muted opacity-0 transition group-hover:opacity-100"
              />
            </button>
          );
        })}
      </div>
    </section>
  );
}
