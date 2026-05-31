import { Filter } from "bad-words";

export type PromptValidation =
  | { ok: true }
  | { ok: false; reason: "empty" | "moderation" };

// Klein is CFG-free, so traditional negative-prompts don't apply. The input
// filter is the only visible enforcement layer we have. Don't echo matched
// terms back to the user.
const filter = new Filter();

export function validatePrompt(prompt: string): PromptValidation {
  const trimmed = prompt.trim();
  if (trimmed.length === 0) return { ok: false, reason: "empty" };
  if (filter.isProfane(trimmed)) return { ok: false, reason: "moderation" };
  return { ok: true };
}

export const MODERATION_REJECT_MESSAGE =
  "This prompt can't be generated. Please try a different one.";
