# Dreaming

This repository is the source for a **long-form, interactive research article**. It follows the same Astro + MDX + React setup as [gpu-utilization-blog](https://github.com/dcunhrya/gpu-utilization-blog).

The published site is a **single scrolling page** with a table of contents, prose sections, and embedded visuals you can explore.

## Live site

**[https://dcunhrya.github.io/dreaming-world-model-blog/](https://dcunhrya.github.io/dreaming-world-model-blog/)**

*(If the link is not live yet, enable GitHub Pages with Source = GitHub Actions in repo settings.)*

## Authors

- **Ryan D'Cunha** — Stanford University

## About the implementation

The article is authored as **MDX** (Markdown with interactive components), built as a static site with **Astro**, and styled with **Tailwind CSS**. Charts and other interactive pieces use **React** where needed. The same content powers both local development and the GitHub Pages deployment in `.github/workflows/`.

If you are looking to **edit** the article or **run** the project locally, the section files live under `src/content/sections/`; the entry page is `src/pages/index.astro`.

## Local development

```bash
npm install
npm run dev
```

Build for production:

```bash
npm run build
npm run preview
```

Replace `public/blog_cover.png` with your article cover image before publishing.
