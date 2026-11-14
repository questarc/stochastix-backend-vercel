# Stochastix AI Backend

FastAPI backend for Stochastix AI.

## Deploying to Vercel

Import this repository as a separate Vercel project with the repository root as the project root. The included `vercel.json` packages the runtime `config/` and `data/` files and sets a 300-second function duration.

Set `NVIDIA_API_KEY` in the Vercel project's Production environment variables for the configured NVIDIA LLM provider. Add any other provider credentials only if you intend to use those providers. Never commit API keys.

This deployment variant omits voice/audio generation and video-generation endpoints or UI to reduce runtime dependency and storage requirements. Piper TTS is available in local Python development environments through `requirements-dev.txt`, but is not installed in the Docker or Vercel deployment.

Install local test dependencies with `pip install -r requirements-dev.txt`; Vercel installs only `requirements.txt`.
The PyTorch LSTM and yfinance dependencies are isolated in `requirements-predictive-pricing.txt` to keep them out of
the Vercel function bundle. Run `pip install -r requirements-predictive-pricing.txt` to enable local predictions;
Docker installs this optional file. To enable forecasts on Vercel, deploy the Docker backend (or another compatible
pricing service) separately and set `PREDICTIVE_PRICING_API_URL` on the Vercel backend to that service's base URL.
Without it, the prediction endpoint returns HTTP 503.

The frontend Vercel project proxies `/api/*` to this backend. Update its rewrite destination to the backend's actual Vercel domain after creating the new project.

## Vercel runtime considerations

Vercel runs FastAPI as serverless functions, not as the Docker container in `Dockerfile`. The existing upload, order, chat, and voice stores are in memory and are not durable or shared across function instances.
