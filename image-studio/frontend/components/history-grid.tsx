"use client";

import { History as HistoryIcon, Trash2 } from "lucide-react";
import type { HistoryEntry } from "@/lib/use-history";
import { resolutionById } from "@/lib/resolutions";
import { cn } from "@/lib/utils";

interface HistoryGridProps {
  entries: HistoryEntry[];
  onSelect: (entry: HistoryEntry) => void;
  onClear: () => void;
  onRemove: (id: string) => void;
  activeEntryId: string | null;
}

export function HistoryGrid({
  entries,
  onSelect,
  onClear,
  onRemove,
  activeEntryId,
}: HistoryGridProps) {
  const hasEntries = entries.length > 0;

  return (
    <section className="rounded-[1.75rem] border border-border-strong bg-surface-raised p-5 shadow-[var(--panel-shadow)] backdrop-blur-xl">
      <header className="flex items-center justify-between gap-3 pb-3">
        <div className="flex items-center gap-3">
          <span className="flex size-11 items-center justify-center rounded-full bg-accent-soft text-accent-strong">
            <HistoryIcon className="size-5" />
          </span>
          <span>
            <span className="block text-lg font-semibold tracking-[-0.03em] text-foreground">Session history</span>
            <span className="block text-xs text-muted">
              {hasEntries ? `${entries.length} recent ${entries.length === 1 ? "render" : "renders"} (max 250)` : "No renders yet this session"}
            </span>
          </span>
        </div>
        {hasEntries ? (
          <button
            type="button"
            onClick={onClear}
            className="flex items-center gap-1.5 rounded-full border border-border-strong bg-surface-raised px-3 py-1.5 text-xs font-medium text-muted-strong backdrop-blur-md transition hover:border-danger/50 hover:text-[#fca5a5]"
          >
            <Trash2 className="size-3.5" />
            Clear
          </button>
        ) : null}
      </header>

      <div className="border-t border-border/70 pt-4">
        {hasEntries ? (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5 2xl:grid-cols-6">
            {entries.map((entry) => {
              const resolution = resolutionById(entry.params.resolutionId);
              const isActive = entry.id === activeEntryId;
              return (
                <div
                  key={entry.id}
                  className={cn(
                    "group relative overflow-hidden rounded-[1.25rem] border bg-surface-strong text-left transition",
                    isActive
                      ? "border-accent shadow-[0_0_0_1px_var(--accent)]"
                      : "border-border-strong hover:border-accent/50",
                  )}
                >
                  <button
                    type="button"
                    onClick={() => onSelect(entry)}
                    className="block w-full text-left"
                    title={entry.prompt || "Untitled prompt"}
                  >
                    <div className="aspect-square w-full overflow-hidden">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        alt={entry.prompt || "Generated image"}
                        src={entry.imageUrl}
                        className="h-full w-full object-cover transition group-hover:scale-[1.03]"
                      />
                    </div>
                    <div className="space-y-1 p-2.5">
                      <p className="line-clamp-2 text-xs leading-4 text-foreground">
                        {entry.prompt || <span className="italic text-muted">untitled</span>}
                      </p>
                      <p className="font-mono text-[10px] tracking-wide text-muted">
                        {resolution.width}×{resolution.height}
                      </p>
                    </div>
                  </button>
                  <button
                    type="button"
                    aria-label="Delete from history"
                    onClick={(e) => {
                      e.stopPropagation();
                      onRemove(entry.id);
                    }}
                    className="absolute right-2 top-2 flex size-7 items-center justify-center rounded-full border border-border-strong bg-surface/90 text-muted-strong opacity-0 backdrop-blur-md transition hover:border-danger/50 hover:text-[#fca5a5] focus-visible:opacity-100 group-hover:opacity-100"
                  >
                    <Trash2 className="size-3.5" />
                  </button>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="rounded-[1.15rem] border border-dashed border-border px-4 py-6 text-center text-sm text-muted">
            Last 250 renders appear here.
          </p>
        )}
      </div>
    </section>
  );
}
