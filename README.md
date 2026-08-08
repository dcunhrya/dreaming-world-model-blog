# Learning to Dream: Train-Time Compute with World Models

This repository is the source for a **long-form, interactive research article** on dreaming in reinforcement learning — train-time compute spent inside learned world models, demonstrated through a **synthetic ED chest-pain diagnostic workup**.

It follows the same Astro + MDX + React setup as [gpu-utilization-blog](https://github.com/dcunhrya/gpu-utilization-blog).

The published site is a **single scrolling page** with a table of contents, prose sections, and embedded visuals you can explore.

## Live site

**[https://dcunhrya.github.io/dreaming-world-model-blog/](https://dcunhrya.github.io/dreaming-world-model-blog/)**

## Authors

- **Ryan D'Cunha** — Stanford University

## Article outline

1. Introduction — real experience is expensive; dreaming is cheap imagined experience
2. The algorithmic problem — MDPs, Q-learning, learned P̂ and R̂
3. What is a world model? — consequence models, not video generators
4. What is dreaming? — imagined rollouts as a train-time compute knob
5. A diagnostic healthcare world model — synthetic chest-pain workup exercise
6. Diagnostic results — sample efficiency, safety, rare PE, model bias
7. The promise of dreaming — web agents, robotics, healthcare bridge
8. Research progression — Dyna → Dreamer → MuZero → foundation world models
9. Conclusion
10. References

## GridWorld branch

The original **GridWorld Dyna-Q** demo (maze navigation, Q-heatmaps, transition-noise failure mode) lives on the [`gridworld-dynaq`](https://github.com/dcunhrya/dreaming-world-model-blog/tree/gridworld-dynaq) branch.

## Implementation

- **Diagnostic engine:** `src/dreaming/diagnostic/` — synthetic chest-pain simulator, empirical world model, Dyna-Q training, baselines
- **Interactive demo:** `DiagnosticWorkupExplorer` — case walkthrough with dream-budget and model-bias sliders
- **Static charts:** `npm run generate:results` → `public/data/diagnostic_results.json`
- **Content:** MDX sections in `src/content/sections/`

### Agent notes

Commit authorship is governed by `.cursor/rules/git-commits.mdc`.

## Local development

```bash
npm install
npm run generate:results   # optional: regenerate chart JSON
npm run test
npm run dev
```

Build for production:

```bash
npm run build
npm run preview
```

Replace `public/blog_cover.png` with your article cover image before publishing.
