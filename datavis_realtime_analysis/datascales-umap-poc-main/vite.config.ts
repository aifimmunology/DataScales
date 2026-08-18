import { defineConfig, type PluginOption, type ViteDevServer } from 'vite'
import react from '@vitejs/plugin-react'
import express from 'express'
import { existsSync } from 'fs'

const DATA_DIR = process.env.DATA_DIR

// Convert gs://bucket/path → https://storage.googleapis.com/bucket/path.
// For local paths, return null so the dev server serves data instead.
function toHttpUrl(dir?: string): string | null {
  if (!dir) return null
  if (dir.startsWith('gs://')) {
    return 'https://storage.googleapis.com/' + dir.slice('gs://'.length)
  }
  return null
}

const DATA_BASE_URL = toHttpUrl(DATA_DIR)

function serveDataDir(dir?: string): PluginOption {
  return {
    name: 'serve-data-dir',
    apply: 'serve',
    configureServer(server: ViteDevServer) {
      if (!dir) {
        server.config.logger.warn(
          '[serve-data-dir] DATA_DIR is not set — no zarr data will be served locally.',
        )
        return
      }
      if (toHttpUrl(dir)) {
        server.config.logger.info(
          `[serve-data-dir] GCS path detected — zarr will be fetched directly from ${toHttpUrl(dir)}`,
        )
        return
      }
      if (!existsSync(dir)) {
        server.config.logger.warn(`[serve-data-dir] DATA_DIR does not exist: ${dir}`)
      }
      const serve = express.static(dir, {
        dotfiles: 'allow',
        setHeaders: (res, filePath) => {
          if (/\.(zarray|zattrs|zgroup)$/.test(filePath)) {
            res.setHeader('Content-Type', 'application/json')
          }
        },
      })
      server.middlewares.use(serve)

      // A missing Zarr resource MUST 404 — not fall through to Vite's SPA
      // index.html. A missing chunk is normal (an all-fill-value column, e.g. a
      // single-category obs like AIFI_L1 on a pure-lineage subset, writes no
      // chunk file); zarrita treats a 404 as "read the fill value", but a 200
      // index.html gets zstd-decoded as chunk bytes and throws. This runs only
      // for requests express.static didn't already serve.
      server.middlewares.use((req, res, next) => {
        const p = (req.url ?? '').split('?')[0]
        if (/\/(zarr|groups)\.json$/.test(p) || /\/c\/[0-9/]+$/.test(p)) {
          res.statusCode = 404
          res.end()
          return
        }
        next()
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), serveDataDir(DATA_DIR)],
  define: {
    // Expose the resolved base URL to the client. For GCS paths this is the
    // https://storage.googleapis.com/… URL; for local paths it's undefined so
    // the client falls back to window.location (dev server serves the data).
    __DATA_BASE_URL__: DATA_BASE_URL ? JSON.stringify(DATA_BASE_URL) : 'undefined',
    // Raw DATA_DIR (local path or gs:// URL) — recorded as the `store` provenance
    // field when exporting a cell selection. Empty string if unset.
    __STORE_ID__: JSON.stringify(DATA_DIR ?? ''),
  },
  optimizeDeps: {
    include: ['@deck.gl/core', '@deck.gl/layers', '@deck.gl/react', 'zarrita'],
    esbuildOptions: { target: 'esnext' },
  },
  server: {
    port: 3000,
    open: true,
    proxy: { '/api': 'http://localhost:8000' },
  },
  build: { target: 'esnext', outDir: 'dist', sourcemap: true },
})
