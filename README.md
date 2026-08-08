# Learning to Dream: Train-Time Compute with World Models

This repository is the source for a **long-form, interactive research article** on dreaming in reinforcement learning — train-time compute spent inside learned world models.

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
5. Toy exercise — live Dyna-Q GridWorld with Q-value heatmaps
6. Results — sample efficiency vs compute efficiency
7. When dreaming fails — model exploitation and transition noise
8. Diagnostic workup — synthetic ED chest-pain simulator (information-state MDP)
9. Diagnostic results — accuracy, rare PE, model bias, cost–safety plots
10. The promise of dreaming — web agents, robotics, healthcare bridge
11. Research progression — Dyna → Dreamer → MuZero → foundation world models
12. Conclusion
13. References

## Implementation

- **GridWorld engine:** TypeScript under `src/dreaming/` (GridWorld, Q-learning, Dyna-Q)
- **Diagnostic engine:** `src/dreaming/diagnostic/` — synthetic chest-pain workup, empirical world model, baselines
- **Interactive demos:** `DynaQExplorer`, `DiagnosticWorkupExplorer`
- **Static charts:** `npm run generate:all-results` → `public/data/dyna_results.json`, `public/data/diagnostic_results.json`
- **Content:** MDX sections in `src/content/sections/`

### Agent notes

Commit authorship is governed by `.cursor/rules/git-commits.mdc`.

## Local development

```bash
npm install
npm run generate:all-results   # optional: regenerate chart JSON
npm run test
npm run dev
```

Build for production:

```bash
npm run build
npm run preview
```

Replace `public/blog_cover.png` with your article cover image before publishing.
