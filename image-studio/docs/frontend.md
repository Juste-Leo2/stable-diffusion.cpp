# `frontend/` — Next.js studio client

Next.js 16 (App Router) + Tailwind 4 + Radix UI client for the four-backend Mac/GPU pipeline. Talks to `backend/` over HTTP at `http://localhost:8000` by default.

## Layout

```
frontend/
    app/                       # Next.js App Router pages
        page.tsx               # Studio (single-prompt → single-image)
        api/                   # Next.js route handlers (proxy to backend)
        globals.css
        layout.tsx
    components/
        studio-client.tsx      # Main interactive shell
        compare-client.tsx     # Three-arm side-by-side (POST /generate/compare)
        result-panel.tsx       # Right-rail result + metadata chips
        history-grid.tsx       # Generated-image history (localStorage-backed)
        bonsai-background.tsx  # Decorative SVG background, theme-aware tint
        theme-toggle.tsx       # Dark/light toggle (next-themes)
        providers.tsx          # Theme + history providers
        ui/                    # Radix-based primitives (Collapsible, etc.)
    lib/
        backends.ts            # GET /backends + the typed client
        use-backends.ts        # SWR-style hook around /backends
        use-history.ts         # localStorage-backed history hook
        use-compare-history.ts # Compare-mode history
        compare-presets.ts     # Preset triples for /generate/compare
        resolutions.ts         # Resolution tier table (mirrors apple's enum)
        utils.ts               # className merging, tiny utils
    public/                    # Static assets (Bonsai SVG, brand)
    tokens/                    # Generated design tokens (mirror of repo-root tokens/)
    next.config.ts             # allowedDevOrigins for 127.0.0.1
    tsconfig.json
    package.json
```

## Run

```sh
cd frontend
npm install
npm run dev      # http://localhost:3000
```

Defaults to `http://localhost:8000` for the backend. Override via the route handlers in `app/api/` if you need a different base.

## Build / lint

```sh
npm run build
npm run lint
```

## Design tokens

Tokens live at the repo root in `tokens/design-tokens.json` and are emitted into `frontend/tokens/` by `scripts/gen-design-tokens.py`. Run that script after editing the JSON.

## Backend probe

The `/backends` cache TTL is 30 s server-side. The frontend's `useBackends` hook caches per-page-load; refresh the page after toggling `MFLUX_STUDIO_FORCE_DISABLE_GPU` on the backend.
