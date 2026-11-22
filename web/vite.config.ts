import { fileURLToPath } from 'node:url';
import { defineConfig, searchForWorkspaceRoot } from 'vite';
import react from '@vitejs/plugin-react';

// The Assumptions dialog reads docs/model-assumptions.md from the repository, outside web/.
const assumptions = fileURLToPath(new URL('../docs/model-assumptions.md', import.meta.url));

export default defineConfig({
  plugins: [react()],
  server: { fs: { allow: [searchForWorkspaceRoot(process.cwd()), assumptions] } },
});
