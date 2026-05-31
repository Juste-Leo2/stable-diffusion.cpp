import { describe, expect, it } from "vitest";
import { validatePrompt } from "@/lib/prompt-moderator";

describe("validatePrompt", () => {
  it("accepts neutral prompts", () => {
    const neutral = [
      "A serene Scandinavian woman in her 40s, soft window light",
      "A koi fish gliding through lily-pad covered water",
      "Cinematic landscape of a forest at dawn",
      "Abstract emerald ink swirling through clear water",
      "Hand-lettered storefront sign at golden hour",
    ];
    for (const prompt of neutral) {
      expect(validatePrompt(prompt)).toEqual({ ok: true });
    }
  });

  it("rejects empty / whitespace-only prompts", () => {
    expect(validatePrompt("")).toEqual({ ok: false, reason: "empty" });
    expect(validatePrompt("   \n\t  ")).toEqual({ ok: false, reason: "empty" });
  });

  it("rejects 5 representative profane prompts", () => {
    // Each entry hits a different category from the bad-words default list.
    // Phrasing is plausible image-prompt context to catch bypasses.
    const bad = [
      "a cartoon shit emoji",
      "a fucking sunset over the lake",
      "asshole wearing a tuxedo",
      "bitch holding a bouquet",
      "dickhead in a bonsai garden",
    ];
    for (const prompt of bad) {
      expect(validatePrompt(prompt)).toEqual({ ok: false, reason: "moderation" });
    }
  });
});
