/// <reference types="vite/client" />

// Injected by vite.config.ts — the DATA_DIR the app was launched with (local
// path or gs:// URL), used as the `store` provenance field in selection exports.
declare const __STORE_ID__: string

interface ImportMetaEnv {
  readonly VITE_ZARR_PATH?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
