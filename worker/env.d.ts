// Augments the `Cloudflare.Env` type used by `import { env } from "cloudflare:workers"`.
// Keep in sync with the `d1`/`r2` binding names configured in .openai/hosting.json.
declare global {
  namespace Cloudflare {
    interface Env {
      DB: D1Database;
    }
  }
}

export {};
