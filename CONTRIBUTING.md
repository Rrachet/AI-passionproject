# Contributing to AirCanvas

AirCanvas is an independent engineering project focused on real-time hand interaction, computer vision and practical interface design.

## Before changing code

Please understand which layer you are changing:

- `src/` — browser UI, camera access, hand landmarks and canvas rendering
- `backend/app/` — Python vision and temporal processing
- `backend/training/` — experiments, datasets and model evaluation
- `docs/` — architecture and technical notes

Keep browser interaction responsive. A Python improvement should have a measurable reason to exist rather than simply moving code from JavaScript to Python.

## Development principles

1. Keep the camera experience fast.
2. Prefer small, testable changes.
3. Do not commit secrets, camera recordings or private datasets.
4. Document experimental vision changes.
5. If tracking changes, test both smooth movement and temporary hand-loss cases.
6. Keep the interface restrained and product-focused.

## Pull requests

A useful pull request should explain:

- what changed
- why it changed
- how it was tested
- whether browser tracking behavior changed
- whether Python behavior changed

Screenshots or short recordings are useful for visual changes.

## Commit style

Use concise, descriptive messages such as:

```text
feat: improve fingertip prediction
fix: recover tracking after short frame loss
style: refine camera surface
perf: reduce vision frame overhead
docs: explain temporal tracker
```

AirCanvas is built through an iterative process: **build, test, observe, measure, improve.**
