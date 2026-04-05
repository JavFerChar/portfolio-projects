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

## Future: Route Recommender

Collaborative filtering on user ascent logs. "Climbers who sent X also liked Y." Content-based fallback using route spatial similarity. Would add `POST /recommend` endpoint.

## Future: Route Generator

Given target grade + angle, generate a new valid route. VAE or diffusion model on hold sequences. Would add `POST /generate` endpoint and a "generate route" tab in dashboard.
