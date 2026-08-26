import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
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
