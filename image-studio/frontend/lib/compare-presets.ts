export interface ComparePreset {
  id: string;
  label: string;
  category: "portrait" | "landscape" | "text" | "geometric" | "abstract" | "still-life" | "animal";
  prompt: string;
}

// Hand-picked battery covering varied styles — portrait, landscape, typography,
// geometric, abstract, still-life, animal. Curate with Evan before shipping
// externally; these are seeded to exercise the differences between backends.
export const COMPARE_PRESETS: ComparePreset[] = [
  {
    id: "portrait-studio",
    label: "Studio portrait",
    category: "portrait",
    prompt:
      "A serene Scandinavian woman in her 40s, soft window light, natural skin texture, shallow depth of field, 50mm portrait lens, neutral linen backdrop.",
  },
  {
    id: "landscape-forest",
    label: "Cinematic landscape",
    category: "landscape",
    prompt:
      "A mist-filled Pacific Northwest forest at dawn, shafts of warm sunlight breaking through cedar trunks, moss-covered ground, cinematic wide composition.",
  },
  {
    id: "text-storefront",
    label: "Typography (storefront)",
    category: "text",
    prompt:
      "A charming corner bakery storefront with a hand-lettered wooden sign that reads 'BONSAI PATISSERIE' in vintage serif letters, morning light, warm wood tones.",
  },
  {
    id: "geometric-pattern",
    label: "Geometric pattern",
    category: "geometric",
    prompt:
      "An isometric architectural rendering of a concrete brutalist pavilion with nested cylindrical voids, strong shadows, golden hour, clean line work, minimal palette.",
  },
  {
    id: "abstract-liquid",
    label: "Abstract liquid",
    category: "abstract",
    prompt:
      "An abstract photograph of emerald ink swirling through clear water, suspended droplets, macro detail, high contrast against black background, shallow focus.",
  },
  {
    id: "still-life-ceramic",
    label: "Still life",
    category: "still-life",
    prompt:
      "A still life of weathered ceramic bowls and a single ripe persimmon on a linen cloth, diffused studio light, muted earth tones, large format film aesthetic.",
  },
  {
    id: "animal-koi",
    label: "Animal (koi)",
    category: "animal",
    prompt:
      "A koi fish gliding through lily-pad covered water, ripples dispersing outward, top-down view, sunlight catching orange and white scales, painterly detail.",
  },
];

export const DEFAULT_PRESET_ID = COMPARE_PRESETS[0].id;
