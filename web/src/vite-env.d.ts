/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Optional: the workspace API this build connects to on its own, e.g. https://twin-api.example.com */
  readonly VITE_TWIN_API_URL?: string;
}
