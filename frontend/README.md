# frontend

The dashboard for [agentic-copilot](../README.md) — see the root README for
what this project is, how to run it, and the API it reads from.

```bash
npm install
npm run dev      # http://localhost:5173, expects `make api` running separately
npm test         # vitest
npm run build    # production build, also runs lint first via `make build`
```

Five routes, all reading the live API (no mock data): Overview, Timeline,
Incidents, Incident detail, Evaluation. See `src/pages/` and `src/api/client.js`.
