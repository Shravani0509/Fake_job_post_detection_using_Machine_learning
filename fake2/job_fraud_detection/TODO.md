# JobShield — TODO (Multimodal Production Upgrade)

> Track progress for the multimodal (text + image) fake job post detection app.

## Step 1 — Multimodal backend foundations
- [ ] Add image upload handling and storage folder.
- [ ] Implement CNN-based image embedding extractor (EfficientNet/ResNet) with caching.
- [ ] Implement fusion model training (text TF-IDF + image embeddings -> classifier).
- [x] Add image embedding extractor (CNN backbone) and multimodal inference wrapper (heuristic fusion placeholder).
- [ ] Save/load multimodal artifacts in `trained_models/`.
- [ ] Extend `src/inference.py` to accept optional image and produce:
  - [ ] prediction + confidence
  - [ ] structured reasons (text rules + text top TF-IDF terms)
  - [ ] structured image signals (embedding-based risk summary)

## Step 2 — Backend API + persistence
- [ ] Update Flask routes to accept image file upload: `/predict` and/or `/api/predict`.
- [ ] Add SQLite tables for prediction history.
- [ ] Store each prediction with inputs + results + explanation JSON.
- [ ] Add `/dashboard` data to show real history counts.
- [ ] Add `/analytics` route.

## Step 3 — Frontend/UX upgrade
- [ ] Implement dark/light mode toggle + theme variables in `base.html`.
- [ ] Redesign prediction page (`templates/index.html`) to include:
  - [ ] image upload dropzone
  - [ ] confidence visualization
  - [ ] “Why this result?” structured cards
  - [ ] risk indicators and warning flags
- [ ] Upgrade dashboard (`templates/dashboard.html`) to show history table.
- [ ] Add analytics page with Chart.js.
- [ ] Polish landing page (better hero + sections + illustrations).

## Step 4 — Training + evaluation
- [ ] Add training script that can generate/load image samples.
- [ ] Produce additional multimodal evaluation metrics and plots.

## Step 5 — Deployment artifacts
- [ ] Add `Dockerfile` + `docker-compose.yml`.
- [ ] Update root `README.md` with architecture diagram + screenshots + run/deploy instructions.

## Step 6 — Quality gates
- [ ] Run unit/manual tests for text-only prediction.
- [ ] Validate image upload end-to-end.
- [ ] Update Selenium test coverage for the new UI.

## Completed
- [x] Audit existing codebase and confirm current implementation is text-only (TF-IDF + RandomForest).
- [x] Approve multimodal production upgrade plan.

