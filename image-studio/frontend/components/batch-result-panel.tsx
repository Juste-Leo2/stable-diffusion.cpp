"use client";

import { useCallback } from "react";
import { Download, LoaderCircle, Share2 } from "lucide-react";
import type { HistoryEntry } from "@/lib/use-history";
import { resolutionById } from "@/lib/resolutions";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { ShareButton } from "@/components/result-panel";
import { buildMetadata, downloadWithMetadata } from "@/lib/png-metadata";
import { cn } from "@/lib/utils";

interface BatchResultPanelProps {
  entries: HistoryEntry[];
  selectedIndex: number | null;
  onSelect: (index: number) => void;
  progress: { done: number; total: number } | null;
  isLoading: boolean;
  error: string | null;
  // Aspect to use for pending tiles (no entry yet) — matches the resolution
  // selected when this batch was kicked off.
  pendingAspectRatio: number;
}

export function BatchResultPanel({
  entries,
  selectedIndex,
  onSelect,
  progress,
  isLoading,
  error,
  pendingAspectRatio,
}: BatchResultPanelProps) {
  const selected = selectedIndex !== null ? entries[selectedIndex] ?? null : null;
  const downloadName = selected
    ? `bonsai-${new Date(selected.timestamp).toISOString().replace(/[:.]/g, "-")}.png`
    : "bonsai.png";
  const handleSave = useCallback(async () => {
    if (!selected) return;
    const res = resolutionById(selected.params.resolutionId);
    const meta = buildMetadata(selected, `${res.width}x${res.height}`);
    await downloadWithMetadata(selected.imageBlob, downloadName, meta);
  }, [selected, downloadName]);

  return (
    <section className="space-y-5">
      <div className="relative overflow-hidden rounded-[1.75rem] border border-border-strong bg-surface-raised p-3 shadow-[var(--panel-shadow)] backdrop-blur-xl">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_18%_14%,var(--halo-a),transparent_22%),radial-gradient(circle_at_82%_18%,var(--halo-b),transparent_18%),linear-gradient(180deg,transparent,rgba(0,0,0,0.04))]" />

        {error ? (
          <div className="relative flex min-h-[360px] items-center justify-center overflow-hidden rounded-[1.5rem] bg-surface-strong px-6 text-center backdrop-blur-md xl:min-h-[520px]">
            <Alert variant="destructive" className="max-w-[520px] text-left">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          </div>
        ) : (
          <div className="relative grid grid-cols-2 gap-3 rounded-[1.5rem] bg-surface-strong p-4 backdrop-blur-md">
            {[0, 1, 2, 3].map((i) => {
              const entry = entries[i];
              const isSelected = selectedIndex === i;
              const showSpinner = !entry && isLoading;
              // Each tile sizes to its own image's aspect ratio; pending tiles
              // adopt the active generate-time aspect so the grid doesn't
              // jump as renders fill in.
              const tileAspect = entry
                ? (() => {
                    const r = resolutionById(entry.params.resolutionId);
                    return r.width / r.height;
                  })()
                : pendingAspectRatio;
              return (
                <button
                  key={i}
                  type="button"
                  disabled={!entry}
                  aria-pressed={isSelected}
                  aria-label={entry ? `Select image ${i + 1}` : `Image ${i + 1} pending`}
                  onClick={() => entry && onSelect(i)}
                  style={{ aspectRatio: tileAspect }}
                  className={cn(
                    "relative overflow-hidden rounded-[1.25rem] border bg-surface-raised transition",
                    isSelected
                      ? "border-accent shadow-[0_0_0_2px_var(--accent)]"
                      : "border-border-strong",
                    entry ? "cursor-pointer hover:border-accent/60" : "cursor-default",
                  )}
                >
                  {entry ? (
                    /* eslint-disable-next-line @next/next/no-img-element */
                    <img
                      alt={entry.prompt || "Generated image"}
                      src={entry.imageUrl}
                      className="h-full w-full object-cover"
                    />
                  ) : showSpinner ? (
                    <div className="flex h-full w-full items-center justify-center">
                      <LoaderCircle className="size-6 animate-spin text-accent" />
                    </div>
                  ) : null}
                </button>
              );
            })}
          </div>
        )}
        {progress && progress.done < progress.total ? (
          <p className="relative mt-3 text-center text-xs text-muted">
            Rendering {Math.min(progress.done + 1, progress.total)} / {progress.total}…
          </p>
        ) : null}
      </div>

      {!error ? (
        <div className="flex flex-col gap-4 border-t border-border/80 pt-5 sm:flex-row sm:items-end sm:justify-between">
          <div className="space-y-2">
            <p className="max-w-[44ch] text-sm leading-6 text-foreground">
              {selected?.prompt
                || (isLoading
                  ? "Rendering 4 images — pick one when done to save or share."
                  : "Pick an image to enable Save and Share.")}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {selected ? (
              <>
                <ShareButton entry={selected} imageUrl={selected.imageUrl} downloadName={downloadName} prompt={selected.prompt} />
                <Button size="lg" type="button" onClick={handleSave}>
                  <Download className="size-4" />
                  Save
                </Button>
              </>
            ) : (
              <>
                <Button
                  size="lg"
                  variant="outline"
                  type="button"
                  disabled
                  aria-disabled
                  className="cursor-not-allowed opacity-40"
                >
                  <Share2 className="size-4" />
                  Share
                </Button>
                <Button
                  size="lg"
                  type="button"
                  disabled
                  aria-disabled
                  className="cursor-not-allowed opacity-40"
                >
                  <Download className="size-4" />
                  Save
                </Button>
              </>
            )}
          </div>
        </div>
      ) : null}
    </section>
  );
}
