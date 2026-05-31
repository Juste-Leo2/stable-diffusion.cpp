"use client";

import { startTransition, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, Dice5, LoaderCircle, X } from "lucide-react";
import {
  MODEL_FAMILIES,
  backendFor,
  type ModelFamily,
} from "@/lib/backends";
import { MODERATION_REJECT_MESSAGE, validatePrompt } from "@/lib/prompt-moderator";
import {
  DEFAULT_RESOLUTION_ID,
  RESOLUTIONS,
  STORAGE_KEY_RESOLUTION,
  resolutionById,
} from "@/lib/resolutions";
import { useHistory, type HistoryEntry } from "@/lib/use-history";
import { useBackends } from "@/lib/use-backends";
import { cn } from "@/lib/utils";
import { BatchResultPanel } from "@/components/batch-result-panel";
import { HistoryGrid } from "@/components/history-grid";
import { ResultPanel } from "@/components/result-panel";
import { SiteNav } from "@/components/site-nav";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

type Mode = "single" | "batch";
const MODE_OPTIONS: { value: Mode; label: string }[] = [
  { value: "single", label: "Single" },
  { value: "batch", label: "Batch of 4" },
];

const GENERATE_PATH = "/api/generate";
const BATCH_SIZE = 4;
const RENDER_TIMEOUT_MS = 120_000;
const ABORT_REASON_USER = "user-cancel";
const ABORT_REASON_TIMEOUT = "timeout";

const SAMPLE_PROMPTS = [
  "A bonsai tree in a quiet ceramic studio, soft morning light, shallow depth of field",
  "A tiny moss garden on a windowsill, rain on the glass, warm film photography",
  "A red fox curled beneath paper lanterns, ink wash style, gentle shadows",
  "A glass terrarium city at night, miniature streets, glowing storefronts",
];

async function parseError(response: Response) {
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    const payload = (await response.json().catch(() => ({}))) as { detail?: string };
    if (typeof payload.detail === "string" && payload.detail.length > 0) return payload.detail;
  }
  const text = await response.text().catch(() => "");
  return text || `Request failed with status ${response.status}.`;
}

function coerceInt(raw: string, fallback: number, minimum: number) {
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) ? Math.max(minimum, parsed) : fallback;
}

export function StudioClient() {
  const promptRef = useRef<HTMLTextAreaElement>(null);

  // The relay reports its kind (mlx | gemlite) at /api/backends; the picker
  // shows the 3 model families and resolves (modelFamily, kind) → canonical
  // backend ID at submit time. Switching kinds = restart the relay.
  const { kind, supportedFamilies, defaultFamily } = useBackends();

  const [prompt, setPrompt] = useState("");
  const [seed, setSeed] = useState(42);
  const [steps, setSteps] = useState(4);
  const [resolutionId, setResolutionId] = useState<string>(DEFAULT_RESOLUTION_ID);
  const [selectedFamily, setSelectedFamily] = useState<ModelFamily>(defaultFamily);
  // Sync the picker to the server-resolved default as soon as /api/backends lands.
  // After the first user pick, we only re-snap when the prior selection is no
  // longer in the supported list — otherwise the user's explicit choice persists.
  useEffect(() => {
    if (!supportedFamilies.length) return;
    if (!supportedFamilies.includes(selectedFamily)) {
      setSelectedFamily(defaultFamily);
    }
  }, [supportedFamilies, defaultFamily, selectedFamily]);
  const familyOptions = useMemo(
    () => MODEL_FAMILIES.filter((f) => supportedFamilies.includes(f.value)),
    [supportedFamilies],
  );
  const resolvedBackend = useMemo(
    () => backendFor(kind, selectedFamily)?.value ?? null,
    [kind, selectedFamily],
  );

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<{ stepMs: number; totalMs: number; peakMemoryMb: number | null } | null>(null);
  const [activeEntryId, setActiveEntryId] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const cancelledRef = useRef(false);

  const handleStop = useCallback(() => {
    cancelledRef.current = true;
    abortRef.current?.abort(ABORT_REASON_USER);
  }, []);

  const [mode, setMode] = useState<Mode>("single");
  const [batchEntries, setBatchEntries] = useState<HistoryEntry[]>([]);
  const [selectedBatchIndex, setSelectedBatchIndex] = useState<number | null>(null);
  const [batchProgress, setBatchProgress] = useState<{ done: number; total: number } | null>(null);

  const { entries, push: pushHistory, clear: clearHistory, remove: removeHistory } = useHistory();

  const handleRemoveEntry = useCallback(
    (id: string) => {
      removeHistory(id);
      setActiveEntryId((prev) => (prev === id ? null : prev));
      setBatchEntries((prev) => prev.filter((e) => e.id !== id));
      setSelectedBatchIndex((prev) => {
        if (prev === null) return null;
        const stillExists = batchEntries[prev]?.id !== id;
        return stillExists ? prev : null;
      });
    },
    [removeHistory, batchEntries],
  );

  useEffect(() => {
    if (typeof window === "undefined") return;
    const stored = window.localStorage.getItem(STORAGE_KEY_RESOLUTION);
    if (!stored || !RESOLUTIONS.some((r) => r.id === stored)) return;
    startTransition(() => {
      setResolutionId(stored);
    });
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(STORAGE_KEY_RESOLUTION, resolutionId);
  }, [resolutionId]);

  useEffect(() => {
    const el = promptRef.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${el.scrollHeight}px`;
  }, [prompt]);

  const resolution = useMemo(() => resolutionById(resolutionId), [resolutionId]);
  const generateDisabled = isLoading || prompt.trim().length === 0;

  const handleRandomizeSeed = useCallback(() => {
    setSeed(Math.floor(Math.random() * 2 ** 31));
  }, []);

  const handleSelectEntry = useCallback((entry: HistoryEntry) => {
    setPrompt(entry.prompt);
    setSeed(entry.params.seed);
    setSteps(entry.params.steps);
    setResolutionId(entry.params.resolutionId);
    setActiveEntryId(entry.id);
    setStats(null);
    setError(null);
    setMode("single");
    setBatchEntries([]);
    setSelectedBatchIndex(null);
    setBatchProgress(null);
  }, []);

  const handleSelectSamplePrompt = useCallback((sample: string) => {
    setPrompt(sample);
    setActiveEntryId(null);
    setError(null);
    promptRef.current?.focus();
  }, []);

  const handleGenerate = useCallback(
    async (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (generateDisabled) return;

      const verdict = validatePrompt(prompt);
      if (!verdict.ok && verdict.reason === "moderation") {
        setError(MODERATION_REJECT_MESSAGE);
        return;
      }

      cancelledRef.current = false;
      setIsLoading(true);
      setError(null);
      setStats(null);
      setBatchEntries([]);
      setSelectedBatchIndex(null);
      setBatchProgress(null);

      if (!resolvedBackend) {
        setError("No backend available for the selected family.");
        setIsLoading(false);
        return;
      }
      const backendForRequest = resolvedBackend;
      const generateOne = async (
        effectiveSeed: number,
      ): Promise<
        | { ok: true; entry: HistoryEntry; stats: { stepMs: number; totalMs: number; peakMemoryMb: number | null } }
        | { ok: false; error: string; cancelled?: boolean }
      > => {
        const controller = new AbortController();
        abortRef.current = controller;
        const timeoutId = setTimeout(
          () => controller.abort(ABORT_REASON_TIMEOUT),
          RENDER_TIMEOUT_MS,
        );
        const startedAt = performance.now();
        let response: Response;
        try {
          response = await fetch(GENERATE_PATH, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            signal: controller.signal,
            body: JSON.stringify({
              prompt,
              seed: effectiveSeed,
              steps,
              height: resolution.height,
              width: resolution.width,
              backend: backendForRequest,
            }),
          });
        } catch (err) {
          clearTimeout(timeoutId);
          abortRef.current = null;
          if (controller.signal.aborted) {
            const reason = String(controller.signal.reason ?? "");
            if (reason === ABORT_REASON_USER) return { ok: false, error: "", cancelled: true };
            if (reason === ABORT_REASON_TIMEOUT) {
              return { ok: false, error: "Render timed out after 120 seconds." };
            }
          }
          void err;
          return { ok: false, error: "Could not reach the render backend." };
        }
        try {
          if (!response.ok) return { ok: false, error: await parseError(response) };
          const peakHeader = response.headers.get("X-Peak-Memory-MB");
          const peakMemoryMb = peakHeader ? Number.parseFloat(peakHeader) : NaN;
          const blob = await response.blob();
          const totalMs = performance.now() - startedAt;
          const entry = pushHistory({
            prompt,
            params: { seed: effectiveSeed, steps, backend: backendForRequest, resolutionId },
            imageBlob: blob,
          });
          return {
            ok: true,
            entry,
            stats: {
              stepMs: totalMs / Math.max(steps, 1),
              totalMs,
              peakMemoryMb: Number.isFinite(peakMemoryMb) ? peakMemoryMb : null,
            },
          };
        } finally {
          clearTimeout(timeoutId);
          abortRef.current = null;
        }
      };

      if (mode === "single") {
        const result = await generateOne(seed);
        if (!result.ok) {
          if (!result.cancelled) setError(result.error);
        } else {
          setActiveEntryId(result.entry.id);
          setStats(result.stats);
        }
      } else {
        setBatchProgress({ done: 0, total: BATCH_SIZE });
        const collected: HistoryEntry[] = [];
        let lastStats: { stepMs: number; totalMs: number; peakMemoryMb: number | null } | null = null;
        for (let i = 0; i < BATCH_SIZE; i++) {
          if (cancelledRef.current) break;
          const result = await generateOne(seed + i);
          if (!result.ok) {
            if (!result.cancelled) setError(result.error);
            break;
          }
          collected.push(result.entry);
          lastStats = result.stats;
          setBatchEntries([...collected]);
          setBatchProgress({ done: i + 1, total: BATCH_SIZE });
        }
        if (lastStats) setStats(lastStats);
      }

      setIsLoading(false);
      setBatchProgress(null);
      cancelledRef.current = false;
    },
    [
      generateDisabled,
      mode,
      prompt,
      pushHistory,
      resolution.height,
      resolution.width,
      resolutionId,
      resolvedBackend,
      seed,
      steps,
    ],
  );

  const activeEntry = useMemo(
    () => entries.find((e) => e.id === activeEntryId) ?? null,
    [activeEntryId, entries],
  );
  const downloadName = activeEntry
    ? `bonsai-${new Date(activeEntry.timestamp).toISOString().replace(/[:.]/g, "-")}.png`
    : "bonsai.png";

  return (
    <main className="relative min-h-screen px-4 py-5 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-5xl space-y-8 pb-12">

        {/* Top bar */}
        <SiteNav />

        {/* Heading */}
        <div className="text-center">
          <h1 className="text-3xl font-medium tracking-[-0.04em] text-foreground/80 sm:text-4xl">
            Create stunning images locally with AI
          </h1>
        </div>

        {/* Generate form */}
        <form onSubmit={handleGenerate}>
          <section className="rounded-[1.5rem] border border-border-strong bg-surface-raised p-3 shadow-[var(--panel-shadow)] backdrop-blur-xl">
            <div className="flex flex-col gap-3">
              <Textarea
                ref={promptRef}
                className="min-h-[120px] flex-1 rounded-[1.15rem] border-border-strong bg-surface-strong px-4 py-3 text-sm leading-6 placeholder:text-muted"
                placeholder="Describe what you want to create…"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
              />

              {/* Compact settings row */}
              <div className="flex flex-wrap items-end gap-2 border-t border-border/60 pt-2">
              <fieldset
                disabled={isLoading}
                className="contents disabled:opacity-50"
              >
                {/* Model family */}
                <div className="flex flex-col gap-1">
                  <span className="text-[9px] font-semibold uppercase tracking-[0.18em] text-muted">Model</span>
                  <div className="relative">
                    <select
                      className="h-9 appearance-none rounded-lg border border-border-strong bg-surface-strong pl-3 pr-8 text-xs text-muted outline-none transition focus:border-accent focus:text-foreground disabled:opacity-60"
                      value={selectedFamily}
                      onChange={(e) => setSelectedFamily(e.target.value as ModelFamily)}
                      disabled={isLoading || familyOptions.length === 0}
                      aria-label="Model family"
                    >
                      {familyOptions.map((f) => (
                        <option key={f.value} value={f.value}>{f.label}</option>
                      ))}
                    </select>
                    <ChevronDown aria-hidden className="pointer-events-none absolute right-2 top-1/2 size-3 -translate-y-1/2 text-muted" />
                  </div>
                </div>

                {/* Resolution */}
                <div className="flex flex-col gap-1">
                  <span className="text-[9px] font-semibold uppercase tracking-[0.18em] text-muted">Resolution</span>
                  <div className="relative">
                    <select
                      className="h-9 appearance-none rounded-lg border border-border-strong bg-surface-strong pl-3 pr-8 text-xs text-muted outline-none transition focus:border-accent focus:text-foreground"
                      value={resolutionId}
                      onChange={(e) => setResolutionId(e.target.value)}
                    >
                      <optgroup label="Poster">
                        {RESOLUTIONS.filter((r) => r.tier === "1024").map((r) => (
                          <option key={r.id} value={r.id}>{r.label}</option>
                        ))}
                      </optgroup>
                      <optgroup label="Quick">
                        {RESOLUTIONS.filter((r) => r.tier === "512").map((r) => (
                          <option key={r.id} value={r.id}>{r.label}</option>
                        ))}
                      </optgroup>
                    </select>
                    <ChevronDown aria-hidden className="pointer-events-none absolute right-2 top-1/2 size-3 -translate-y-1/2 text-muted" />
                  </div>
                </div>

                {/* Seed */}
                <div className="flex flex-col gap-1">
                  <span className="text-[9px] font-semibold uppercase tracking-[0.18em] text-muted">Seed</span>
                  <div className="flex items-center gap-1">
                    <Input
                      type="number"
                      min={0}
                      step={1}
                      value={seed}
                      onChange={(e) => setSeed(coerceInt(e.target.value, seed, 0))}
                      className="h-9 w-36 text-xs"
                    />
                    <button
                      type="button"
                      onClick={handleRandomizeSeed}
                      aria-label="Randomize seed"
                      className="flex size-9 items-center justify-center rounded-lg border border-border-strong bg-surface-strong text-muted transition hover:border-accent/60 hover:text-foreground"
                    >
                      <Dice5 className="size-3.5" />
                    </button>
                  </div>
                </div>

                {/* Steps */}
                <div className="flex flex-col gap-1">
                  <span className="text-[9px] font-semibold uppercase tracking-[0.18em] text-muted">Steps</span>
                  <Input
                    type="number"
                    min={1}
                    step={1}
                    value={steps}
                    onChange={(e) => setSteps(coerceInt(e.target.value, steps, 1))}
                    className="h-9 w-20 text-xs"
                  />
                </div>

                {/* Mode toggle */}
                <div
                  role="radiogroup"
                  aria-label="Render mode"
                  className="flex items-center gap-0.5 rounded-full border border-border-strong bg-background/55 p-0.5 text-xs font-medium"
                >
                  {MODE_OPTIONS.map((opt) => {
                    const selected = mode === opt.value;
                    return (
                      <button
                        key={opt.value}
                        role="radio"
                        type="button"
                        aria-checked={selected}
                        disabled={isLoading}
                        onClick={() => setMode(opt.value)}
                        className={cn(
                          "rounded-full px-3 py-1.5 transition-all duration-200 ease-out disabled:cursor-not-allowed disabled:opacity-50",
                          selected
                            ? "bg-cta-bg text-cta-ink shadow-[0_1px_0_0_rgba(255,255,255,0.3)_inset]"
                            : "text-muted hover:text-foreground",
                        )}
                      >
                        {opt.label}
                      </button>
                    );
                  })}
                </div>

              </fieldset>
                {/* Generate / Stop — siblings of fieldset so Stop stays clickable while loading */}
                <div className="ml-auto flex items-center gap-2">
                  <Button
                    className="relative h-11 overflow-hidden px-6 text-sm"
                    disabled={generateDisabled}
                    type="submit"
                  >
                    {isLoading ? (
                      <span className="pointer-events-none absolute inset-0 bg-[linear-gradient(120deg,transparent,rgba(255,255,255,0.25),transparent)] animate-[bonsai-shimmer_2.1s_linear_infinite]" />
                    ) : null}
                    {isLoading ? <LoaderCircle className="relative size-3.5 animate-spin" /> : null}
                    <span className="relative">
                      {isLoading
                        ? mode === "batch" && batchProgress
                          ? `${Math.min(batchProgress.done + 1, batchProgress.total)} / ${batchProgress.total}`
                          : "Generating…"
                        : "Generate"}
                    </span>
                  </Button>
                  {isLoading ? (
                    <button
                      type="button"
                      onClick={handleStop}
                      aria-label="Stop render"
                      className="flex h-11 items-center gap-1.5 rounded-md border border-border-strong bg-surface-strong px-3 text-xs font-medium text-muted-strong transition hover:border-danger/50 hover:text-[#fca5a5]"
                    >
                      <X className="size-3.5" />
                      Stop
                    </button>
                  ) : null}
                </div>
              </div>
            </div>

            <div className="flex flex-wrap gap-2 border-t border-border/70 pt-3 mt-1">
              {SAMPLE_PROMPTS.map((sample) => (
                <button
                  key={sample}
                  type="button"
                  onClick={() => handleSelectSamplePrompt(sample)}
                  className="rounded-full border border-border-strong bg-surface-strong px-3 py-2 text-left text-xs leading-5 text-muted-strong transition hover:border-accent/60 hover:text-foreground"
                >
                  {sample}
                </button>
              ))}
            </div>
          </section>
        </form>

        {/* Result */}
        {mode === "single" ? (
          <ResultPanel
            downloadName={downloadName}
            entry={activeEntry}
            error={error}
            imageUrl={activeEntry?.imageUrl ?? null}
            isLoading={isLoading}
            prompt={activeEntry?.prompt ?? prompt}
            stats={stats}
          />
        ) : (
          <BatchResultPanel
            entries={batchEntries}
            selectedIndex={selectedBatchIndex}
            onSelect={setSelectedBatchIndex}
            progress={batchProgress}
            isLoading={isLoading}
            error={error}
            pendingAspectRatio={resolution.width / resolution.height}
          />
        )}

        <HistoryGrid
          activeEntryId={activeEntryId}
          entries={entries}
          onClear={() => {
            clearHistory();
            setActiveEntryId(null);
            setBatchEntries([]);
            setSelectedBatchIndex(null);
          }}
          onRemove={handleRemoveEntry}
          onSelect={handleSelectEntry}
        />

        <footer className="pt-4 text-[10px] font-medium uppercase tracking-[0.24em] text-muted-strong/60">
          <p className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <a href="https://prismml.com/privacy" target="_blank" rel="noreferrer" className="hover:text-foreground">
              Privacy Policy
            </a>
            <span aria-hidden className="text-muted-strong/40">·</span>
            <a href="https://prismml.com/terms" target="_blank" rel="noreferrer" className="hover:text-foreground">
              Terms of Service
            </a>
            <span aria-hidden className="text-muted-strong/40">·</span>
            <span>© 2026 Prism ML, Inc.</span>
          </p>
        </footer>
      </div>
    </main>
  );
}
