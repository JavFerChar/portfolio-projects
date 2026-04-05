# Roadmap

## Phase 5: Serving (next)

### FastAPI endpoint
- `POST /predict` — accepts hold list + angle, returns predicted grade + V-grade
- `GET /health` — health check
- Pydantic request/response models with input validation
- Model loaded once at startup

### Streamlit dashboard
- Visual board grid where you can select holds
- Angle slider (0-70°)
- Predict button → displays grade + SHAP waterfall for that prediction
- Hold usability heatmap overlay

### Docker
- Dockerfile (multi-stage build)
- docker-compose.yml: API (8000) + Streamlit (8501) + MLflow UI (5000)
- `make docker-up` brings everything up

### Tests
- FastAPI TestClient: valid/invalid requests, health endpoint
- Integration test: predict endpoint returns valid grade

## Phase 6: Production Polish

- GitHub Actions CI: test → lint → docker build
- README: architecture diagram, screenshots/GIF of dashboard, CI badge
- Demo GIF for README (Streamlit → select holds → predict → see grade)
- Data refresh automation: scheduled `boardlib database kilter` sync + retrain pipeline with metric comparison (promote new model only if test MAE improves or holds steady)

## Phase 7: Deploy to Hugging Face Spaces

Deploy the Streamlit dashboard to HF Spaces so anyone with the link can use the grade predictor without installing anything. Free tier, no infra. This is the fastest path to "friends can test it."

## Phase 8: Route Generator

Build a generative model that creates new climbing routes for a target grade and angle. The predictor tells you how hard a route is; the generator creates routes at a desired difficulty.

### Model approach
- Input: target grade (V-grade) + angle (0-70°)
- Output: valid set of hold placements + roles (frames string)
- Architecture options: VAE on hold sequences, diffusion on board grid representation, or autoregressive model (placement by placement)
- Constraint: generated routes must only use holds that physically exist on the board (valid placement_ids)
- Validation: run the grade predictor on generated routes to verify they match the target grade

### Dashboard integration
- "Generate route" tab in Streamlit: select target grade + angle → generate → visualize on board → predict grade as sanity check
- Allow regeneration ("try another") and manual hold editing

## Phase 9: Kilter App Integration

Upload generated routes directly to the Kilter Board app so friends can climb them on a real board.

### How it works
The Aurora Climbing API (`api.kilterboardapp.com/v1/sync`) is bidirectional — the app uses PUT to push new climbs. Nobody has built a public tool for programmatic route creation yet. This would be a first.

### Technical approach
1. Authenticate via `/v1/logins` (username/password → Bearer token)
2. Reverse-engineer the sync PUT payload by intercepting app traffic (mitmproxy)
3. Push generated routes as `draft_climbs` entries (drafts, not public — to avoid polluting the shared database)
4. Route appears in the creator's Kilter Board app ready to share or publish

### Considerations
- No official API documentation exists — all based on reverse engineering
- Use `draft_climbs` (not `climbs`) to avoid spamming the public route database
- Respect the community: only publish routes that have been climbed/validated
- Authentication requires a real Kilter Board account

## Future: Route Recommender

Collaborative filtering on user ascent logs. "Climbers who sent X also liked Y." Content-based fallback using route spatial similarity. Would add `POST /recommend` endpoint. Requires authenticated database sync to access ascent logs (currently empty without auth).
