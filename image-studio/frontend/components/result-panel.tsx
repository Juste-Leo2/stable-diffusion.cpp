"use client";

import { useCallback, useState } from "react";
import { Download, LoaderCircle, Share2 } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import type { HistoryEntry } from "@/lib/use-history";
import { buildMetadata, downloadWithMetadata, injectMetadata } from "@/lib/png-metadata";
import { resolutionById } from "@/lib/resolutions";

interface ResultPanelProps {
  downloadName: string;
  error: string | null;
  imageUrl: string | null;
  isLoading: boolean;
  prompt: string;
  stats: { stepMs: number; totalMs: number; peakMemoryMb: number | null } | null;
  entry: HistoryEntry | null;
}

function formatSeconds(ms: number) {
  return `${(ms / 1000).toFixed(1)}s`;
}

export function ResultPanel({
  downloadName,
  error,
  imageUrl,
  isLoading,
  prompt,
  stats,
  entry,
}: ResultPanelProps) {
  const handleSave = useCallback(async () => {
    if (!entry) return;
    const res = resolutionById(entry.params.resolutionId);
    const meta = buildMetadata(entry, `${res.width}x${res.height}`);
    await downloadWithMetadata(entry.imageBlob, downloadName, meta);
  }, [entry, downloadName]);
  return (
    <section className="space-y-5">
      <div className="relative overflow-hidden rounded-[1.75rem] border border-border-strong bg-surface-raised p-3 shadow-[var(--panel-shadow)] backdrop-blur-xl">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_18%_14%,var(--halo-a),transparent_22%),radial-gradient(circle_at_82%_18%,var(--halo-b),transparent_18%),linear-gradient(180deg,transparent,rgba(0,0,0,0.04))]" />
        <div className="absolute right-5 top-5 hidden opacity-[0.06] sm:block dark:opacity-[0.12]">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            alt=""
            aria-hidden
            src="/brand/bonsai-logo-horizontal-dark.svg"
            className="block h-14 w-auto dark:hidden"
            draggable={false}
          />
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            alt=""
            aria-hidden
            src="/brand/bonsai-logo-stacked-light.svg"
            className="hidden h-24 w-auto dark:block"
            draggable={false}
          />
        </div>

        {error ? (
          <div className="relative flex min-h-[360px] items-center justify-center overflow-hidden rounded-[1.5rem] bg-surface-strong px-6 text-center backdrop-blur-md xl:min-h-[520px]">
            <Alert variant="destructive" className="max-w-[520px] text-left">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          </div>
        ) : isLoading ? (
          <div className="relative flex min-h-[420px] flex-col items-center justify-center overflow-hidden rounded-[1.5rem] bg-surface-strong px-8 text-center backdrop-blur-md xl:min-h-[620px]">
            <div
              aria-hidden
              className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_50%_55%,var(--accent-soft),transparent_58%)]"
            />
            <span
              aria-hidden
              className="block h-32 w-32 bg-accent motion-safe:animate-[bonsai-breathe_2.4s_ease-in-out_infinite]"
              style={{
                WebkitMaskImage: "url('/brand/bonsai-icon-horizontal-dark.svg')",
                maskImage: "url('/brand/bonsai-icon-horizontal-dark.svg')",
                WebkitMaskRepeat: "no-repeat",
                maskRepeat: "no-repeat",
                WebkitMaskSize: "contain",
                maskSize: "contain",
                WebkitMaskPosition: "center",
                maskPosition: "center",
              }}
            />
            <div className="relative mt-8 flex items-center gap-2.5 text-sm font-medium text-foreground">
              <LoaderCircle className="size-4 animate-spin text-accent" />
              <span>Rendering…</span>
            </div>
          </div>
        ) : imageUrl ? (
          <div className="relative flex min-h-[420px] items-center justify-center overflow-hidden rounded-[1.5rem] bg-surface-strong p-5 backdrop-blur-md sm:p-7 xl:min-h-[620px]">
            <div className="absolute inset-x-8 bottom-8 h-14 rounded-full bg-black/10 blur-3xl light:bg-[rgba(38,44,53,0.08)]" />
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              alt={prompt || "Generated image"}
              className="relative z-10 max-h-[min(70vh,900px)] max-w-full rounded-[1.5rem] object-contain shadow-[0_42px_84px_-48px_rgba(0,0,0,0.52)]"
              src={imageUrl}
            />
          </div>
        ) : (
          <div className="relative min-h-[360px] overflow-hidden rounded-[1.5rem] bg-surface-strong backdrop-blur-md xl:min-h-[520px]" />
        )}
      </div>

      {imageUrl ? (
        <div className="flex flex-col gap-4 border-t border-border/80 pt-5 sm:flex-row sm:items-end sm:justify-between">
          <div className="space-y-2">
            <p className="max-w-[44ch] text-sm leading-6 text-foreground">{prompt || "Generated image"}</p>
            <p className="text-xs text-muted">
              total {stats ? formatSeconds(stats.totalMs) : "—"} · avg step {stats ? formatSeconds(stats.stepMs) : "—"}
              {stats?.peakMemoryMb != null ? ` · peak ${stats.peakMemoryMb.toFixed(0)} MB` : ""}
            </p>
          </div>

          <div className="flex items-center gap-2">
            <ShareButton entry={entry} downloadName={downloadName} prompt={prompt} imageUrl={imageUrl} />
            <Button size="lg" type="button" onClick={handleSave} disabled={!entry}>
              <Download className="size-4" />
              Save
            </Button>
          </div>
        </div>
      ) : null}
    </section>
  );
}

export function ShareButton({
  entry,
  imageUrl,
  downloadName,
  prompt,
}: {
  entry: HistoryEntry | null;
  imageUrl: string;
  downloadName: string;
  prompt: string;
}) {
  const [supported, setSupported] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);

  // Probe support at click time rather than render time so SSR stays
  // deterministic (navigator only exists client-side).
  const detectSupport = useCallback(() => {
    if (typeof navigator === "undefined" || typeof navigator.share !== "function") return false;
    if (typeof navigator.canShare !== "function") return false;
    return true;
  }, []);

  const handleShare = useCallback(async () => {
    if (busy) return;
    if (!detectSupport()) {
      setSupported(false);
      return;
    }
    setBusy(true);
    try {
      let blob: Blob;
      if (entry) {
        const res = resolutionById(entry.params.resolutionId);
        const meta = buildMetadata(entry, `${res.width}x${res.height}`);
        blob = await injectMetadata(entry.imageBlob, meta);
      } else {
        blob = await fetch(imageUrl).then((r) => r.blob());
      }
      const file = new File([blob], downloadName, { type: blob.type || "image/png" });
      const data: ShareData = { files: [file], title: prompt || "Bonsai render" };
      if (!navigator.canShare(data)) {
        setSupported(false);
        return;
      }
      await navigator.share(data);
      setSupported(true);
    } catch {
      // User dismissed the share sheet, or the platform refused mid-flight.
      // Either way no follow-up — the Save button is right next to it.
    } finally {
      setBusy(false);
    }
  }, [busy, detectSupport, downloadName, entry, imageUrl, prompt]);

  if (supported === false) return null;

  return (
    <Button size="lg" variant="outline" type="button" onClick={handleShare} disabled={busy}>
      <Share2 className="size-4" />
      Share
    </Button>
  );
}
