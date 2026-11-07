# Stochastix AI Backend

FastAPI backend for Stochastix AI.

## Deploying to Vercel

Import this repository as a separate Vercel project with the repository root as the project root. The included `vercel.json` packages the runtime `config/`, `data/`, and downloaded Piper voice files, downloads the configured voice during the build, and sets a 300-second function duration.

Set `NVIDIA_API_KEY` in the Vercel project's Production environment variables for the configured NVIDIA LLM provider. Add any other provider credentials only if you intend to use those providers. Never commit API keys.

This deployment variant has voice/audio summaries but no video-generation endpoints or UI. Its production dependencies omit Matplotlib, FFmpeg, pytest, and Uvicorn's optional standard extras to reduce the Python function bundle. Confirm the final size in Vercel's deployment output; the standard function limit is 500 MB.

Install local test dependencies with `pip install -r requirements-dev.txt`; Vercel installs only `requirements.txt`.

The frontend Vercel project proxies `/api/*` to this backend. Update its rewrite destination to the backend's actual Vercel domain after creating the new project.

## Vercel runtime considerations

Vercel runs FastAPI as serverless functions, not as the Docker container in `Dockerfile`. The existing upload, order, chat, and voice stores are in memory and are not durable or shared across function instances.
