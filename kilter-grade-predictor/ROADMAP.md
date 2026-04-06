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

## Phase 8: Agentic Route Generator

An LLM agent that creates climbing routes for a target grade and angle by iteratively proposing holds and using our grade predictor as a validation tool. Two implementations to demonstrate breadth.

### Agent loop
1. **Propose** — select initial holds based on target grade (use hold usability scores to pick appropriate difficulty)
2. **Predict** — call grade predictor tool → get predicted grade for current hold set
3. **Evaluate** — is predicted grade within tolerance of target? are holds valid?
4. **Refine** — reason about what to change ("route is too easy, swap the jug at position X for a nearby crimp")
5. **Repeat** until grade matches target or max iterations reached

### Agent tools
- `predict_grade(holds, angle)` — our trained XGBoost model
- `get_valid_holds()` — list of valid placement_ids with x, y coordinates
- `get_hold_usability(placement_id)` — per-hold difficulty score
- `validate_route(holds)` — check start/finish presence, hold reachability (max_move_dist)

### Two implementations
**Claude API + tool_use** — direct Anthropic SDK, minimal abstractions. The agent receives the tool definitions, reasons about hold selection in natural language, and makes structured tool calls. Shows understanding of the raw API and tool_use protocol.

**LangChain/LangGraph** — same tools wrapped as LangChain tools, agent orchestrated via LangGraph state machine. Shows familiarity with the industry-standard orchestration framework. Enables easier comparison of different LLM backends.

Both implementations share the same tool functions and evaluation framework.

### Dashboard integration
- "Generate route" tab in Streamlit: select target grade + angle → agent generates → visualize on board → show predicted grade + reasoning trace
- Allow regeneration ("try another") and manual hold editing

## Phase 8b: Agent Evaluation Framework

Systematic benchmarking of route generation quality across both agent implementations and a random baseline.

### Metrics (5 categories)

**1. Grade accuracy**
- `|target_grade - predicted_grade|` per route
- Success rate: % of routes within 1 grade step of target
- Breakdown by grade bucket (easier grades should be more accurate)

**2. Convergence efficiency**
- Tool calls per successful route generation
- Compare: Claude API agent vs LangChain agent vs random baseline (random hold sampling until grade matches)
- An agent that reasons should converge in ~5-10 iterations vs ~100+ for random

**3. Route validity**
- Has start hold(s) and finish hold(s)
- Only uses valid placement_ids on the 12x12 board
- Holds are reachable: `max_move_dist` within reasonable bounds (< 80 board units)
- No duplicate holds

**4. Route naturalness**
- Compute spatial feature distributions of generated routes vs real routes at the same grade
- KL divergence or Wasserstein distance between feature distributions
- A good agent produces routes statistically similar to human-set routes

**5. Diversity**
- Unique hold sets across N generated routes for the same target
- Spatial spread: convex hull area variance across generated routes
- Agent should not converge to the same route repeatedly

### Benchmark protocol
Generate 100 routes per grade bucket (V0-V14) at 3 angles (20°, 40°, 60°). Report all 5 metrics for each (agent implementation, grade, angle) combination. Store results in MLflow as a separate experiment.

## Phase 8c: LLM Observability & Tracing

Production-grade monitoring for the agent — track every reasoning step, tool call, and outcome.

### Tracing stack
- **LangSmith** for the LangChain agent — full trace visualization, tool call sequences, latency per step
- **Braintrust** or **Logfire** for the Claude API agent — structured logging of messages, tool calls, and responses
- Both feed into a unified dashboard

### What to track per generation
- Token usage (input/output) and cost
- Latency (total and per tool call)
- Full tool call sequence (which tools called in what order)
- Agent reasoning text at each step
- Success/failure and final grade accuracy

### Dashboard
- Agent success rate over time (rolling window)
- Average tool calls per route by grade bucket
- Cost per generation (tokens × price)
- Failure mode breakdown (grade miss, invalid route, max iterations, etc.)
- Trace viewer: click any generation to see the full reasoning chain

### Why this matters
LLM observability is a top hiring signal for AI engineering roles. It demonstrates you can debug, monitor, and optimise production AI systems — not just build demos.

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

## Phase 10: Conversational Route Explorer (stretch)

An LLM agent that helps climbers find routes through natural language queries. Uses the same observability stack as the generator.

### Example interactions
- "Find me a V5 at 40° that's more lateral than vertical"
- "Show me routes similar to 'Bell of the Wall' but one grade harder"
- "What are the most popular V3 routes at 30°?"

### Agent tools
- `query_routes(grade_range, angle, min_ascents)` — filter the database
- `predict_grade(holds, angle)` — grade prediction
- `compute_route_similarity(route_a, route_b)` — spatial feature distance
- `visualize_route(holds, angle)` — render route on board image

### Same observability
LangSmith/Braintrust tracing, token tracking, query latency monitoring.

## Future: Route Recommender

Collaborative filtering on user ascent logs. "Climbers who sent X also liked Y." Content-based fallback using route spatial similarity. Would add `POST /recommend` endpoint. Requires authenticated database sync to access ascent logs (currently empty without auth).
