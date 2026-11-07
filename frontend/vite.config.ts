import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The frontend always calls "/api/..." — in dev Vite proxies it to the local
// backend; in Docker nginx does the same. No baked-in backend URL either way.
export default defineConfig({
  plugins: [react()],
  // VITE_FIREBASE_* come from the repo-root .env, shared with docker-compose.
  // Only VITE_-prefixed values reach the browser bundle.
  envDir: '..',
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
