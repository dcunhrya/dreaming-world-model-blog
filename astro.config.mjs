import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import react from '@astrojs/react';
import tailwind from '@astrojs/tailwind';

// GitHub Pages: set SITE_URL and BASE_PATH in CI (see .github/workflows/deploy-github-pages.yml).
// Local dev: omit both; defaults keep base at "/" and work on localhost.
const site = process.env.SITE_URL;
let base = process.env.BASE_PATH ?? '/';
if (base !== '/' && !base.endsWith('/')) base = `${base}/`;
if (base !== '/' && !base.startsWith('/')) base = `/${base}`;

export default defineConfig({
  ...(site ? { site } : {}),
  base,
  integrations: [
    mdx({
      syntaxHighlight: 'shiki',
      gfm: true,
    }),
    react(),
    tailwind({
      applyBaseStyles: false,
    }),
  ],
});
