# Parallel · Financial Twin UI

React + TypeScript + Vite frontend for Vercel. All simulation and LLM work runs in
`twin/api.py` on your laptop. The browser connects directly to FastAPI over HTTPS.
No Python model, provider credentials, or private persona data is deployed to Vercel.

Deployment destination is pending selection of the intended Vercel workspace.
Do not deploy this project to Loki. Git-based automatic deployment is not configured.

## Run locally

From the repository root:

```bash
python -m pip install -e '.[api]'
cp .env.example .env
```

Fill in `.env` locally (never commit it):

- `TWIN_API_KEY`: a random access key of at least 24 characters. Generate one with
  `python -c "import secrets; print(secrets.token_urlsafe(32))"` in your terminal.
- `TOGETHER_API_KEY`: your Together AI key. All LLM calls go to Together AI.
- `TOGETHER_MODEL`: a chat model id from your Together account
  (https://api.together.ai/models). There is no hardcoded model ID. The openai Python client
  talks to Together's OpenAI-compatible API.
- `TOGETHER_BASE_URL`: optional, defaults to `https://api.together.xyz/v1`.
- `TWIN_ALLOWED_ORIGINS`: comma-separated, exact frontend origins, with no trailing slash.

Then start the laptop API with one worker (sessions live in that worker’s memory):

```bash
python -m uvicorn twin.api:app --host 127.0.0.1 --port 8000 --env-file .env
```

The trained artifacts in `twin/artifacts/` and the repository’s data files must be
available. Model artifacts are loaded lazily on the first exploration. In another terminal:

```bash
cd web
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. Choose **Connect workspace**, enter
`http://127.0.0.1:8000` and your **workspace** access key, then **Explore this path**.
Do not enter your provider API key into the website.

## Connect the Vercel UI to your laptop

1. Keep FastAPI running on port 8000.
2. Start an HTTPS tunnel to it. For example, if Cloudflare Tunnel is installed:

   ```bash
   cloudflared tunnel --url http://127.0.0.1:8000
   ```

3. Add the actual Vercel origin to `TWIN_ALLOWED_ORIGINS` in your laptop’s `.env`.
   Keep the localhost entries if you still use the local UI. Restart FastAPI after changes.

   ```dotenv
   TWIN_ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173,https://YOUR-APP.vercel.app
   ```

4. In the Vercel UI, choose **Connect workspace** and enter the HTTPS tunnel URL
   and `TWIN_API_KEY`. The key is held only in tab memory and cleared by a page reload.
5. Build your persona, select a possibility, and choose **Explore this path**.

The laptop and tunnel must stay awake and connected. A hosted HTTPS page needs an
HTTPS backend URL. A Vercel server cannot reach your laptop via its own `localhost`.
Quick tunnel URLs change on restart; reconnect in the UI with the new URL.
Any HTTPS tunnel that forwards Authorization headers and CORS requests can be used.

## Deploy

Import the repository into Vercel and set **Root Directory** to `web`, **Framework**
to Vite, **Build Command** to `npm run build`, and **Output Directory** to `dist`.
The checked-in `vercel.json` also declares these build settings. Vercel does not need
any LLM or API secrets; connection details are supplied through the UI.

The initial screen is an explicitly labeled example: Mia’s fictional profile,
500 actual simulation paths, a CHF 4,000 part-time offer, and an illustrative
conversation script. It does not call an LLM or the laptop until connected.

## Interaction and data flow

```text
Persona builder + scenario
    → POST /sessions (Bearer workspace key)
    → validated custom person + existing World / RulePolicy
    → baseline + paired scenario with shared random draws
    → summarized evidence → LLM explanation
    → chart, metrics, and conversational response

Follow-up question
    → POST /sessions/{id}/messages
    → server-held evidence + bounded conversation history → LLM response

Your plan (own words) or a change of plans
    → POST /plans or POST /plans/{id}/nudges  → 202, a planning job on the laptop
    → GET /plans/{id} every 1.5 s: progress, then the plan and how its steps played out
    → POST /sessions with plan_id: the new plan is explored and explained
```

The planner agent writes the person's plan as yearly steps with fallbacks and tests it
in 300 simulated lives (one to three minutes). A nudge revises only what the change
affects. While "Follow my plan" is on, explorations follow the latest version; editing
the persona detaches the plan until the user starts over. Without a connection the page
shows a real planner run for the example persona (`src/example-plan.json`).

Changing a persona or scenario marks the results as old and disables follow-up
chat until the user runs the new exploration. Follow-ups discuss the existing run;
they cannot silently alter input values or run unrequested scenarios. Numerical
results come from the simulator; the LLM is instructed to explain only that evidence.
Generated explanations still require human judgment and may contain mistakes.

## Model boundaries

- The existing model starts in January **2025**, ends in December **2034**, and
  checks the primary goal in December **2029**. This UI preserves those semantics.
  It is not a rolling forecast beginning today. Enter age and balances at the model start.
- Persona preferences update all eight existing rule-policy traits. Four preferences
  are shown first; the remaining four are available under fine-tuning.
- Part-time work is a salary offer at the opening review. Spending and housing
  changes are one-time intentions at that review. Later annual decisions can change.
- Housing cannot go below the model’s minimum tier for the household. Part-time
  offers are rejected at or beyond the model’s forced retirement age.
- Fan bands are path percentiles; the comparison interval covers Monte Carlo
  sampling error, not real-world calibration or all model uncertainty.
- This frontend creates fresh explorations. It does not modify the existing
  Streamlit journal or implement annual check-ins.

## Privacy and operation

The API fails closed without a sufficiently long workspace key. Configure exact CORS
origins. This is a shared-key research workspace, not a multi-user banking service.
The provider receives relevant persona details (name excluded), simulation summaries,
and conversation messages; user-entered names in messages are not automatically removed.
The frontend keeps persona drafts and connection details in memory only.
Sessions and plans expire after one hour, are removed on subsequent access/creation, and are
bounded to 32 each in memory. The planner receives the person's own words about their plans. Restarting the API clears all sessions. A new successful
exploration requests deletion of the previous conversation. No chat is written to disk.
Two concurrent model/LLM requests are allowed; extra requests receive a retryable 429.
Provider requests have a 45-second timeout and no automatic paid retry.

## Verification

```bash
# From the repository root. Disable unrelated globally installed pytest plugins.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q

cd web
npm run build
npx playwright test
```

Browser tests use installed Google Chrome (`channel: chrome`); install Chrome or
adjust the Playwright channel for your environment. They exercise desktop/mobile
persona creation, chart controls, connection errors, and conversation UI with a
mock API. Python tests exercise actual simulations and API boundaries with a mock
LLM. A live provider call requires your own configured credentials.

References: [Vite on Vercel](https://vercel.com/docs/frameworks/frontend/vite),
[FastAPI CORS](https://fastapi.tiangolo.com/tutorial/cors/).
