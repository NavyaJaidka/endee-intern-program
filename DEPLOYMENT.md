# Deployment Guide

This guide explains how to deploy the AI Research & Code Copilot project with the backend on **Render** and the frontend on **Vercel**.

## Phase 1: Deploy Backend on Render

1.  **Connect GitHub**: Log in to [Render](https://render.com/) and connect your GitHub repository.
2.  **Create New Blueprint**: Click "New" -> "Blueprint".
3.  **Select Repository**: Select this repository. Render will automatically detect the `render.yaml` file.
4.  **Configure Environment Variables**: In the Render dashboard, go to the "endee-backend" service and add the following:
    - `OPENAI_API_KEY`: Your OpenAI API key (or `ANTHROPIC_API_KEY`).
    - `LLM_PROVIDER`: Set to `openai` or `anthropic`.
    - `ENDEE_BASE_URL`: The URL of your hosted Endee instance.
5.  **Deploy**: Click "Apply" to start the deployment. Once finished, copy the **Backend URL** (e.g., `https://endee-backend.onrender.com`).

## Phase 2: Deploy Frontend on Vercel

1.  **Connect GitHub**: Log in to [Vercel](https://vercel.com/) and import this repository.
2.  **Configure Project**:
    - **Framework Preset**: Vite.
    - **Root Directory**: `frontend`.
3.  **Environment Variables**: Add a new environment variable:
    - `VITE_API_URL`: Your Render Backend URL (e.g., `https://endee-backend.onrender.com/api/v1`).
4.  **Deploy**: Click "Deploy". Vercel will build and serve your frontend.

## Verifying the Deployment

1.  Open your Vercel URL.
2.  The frontend should now make requests to your Render backend.
3.  Check the "Health" or "Status" indicators in the UI to confirm connectivity.

---

### Troubleshooting

- **CORS Errors**: If you encounter CORS issues, update `backend/main.py` to include your Vercel domain in the `allow_origins` list.
- **Endee Connection**: Ensure your `ENDEE_BASE_URL` is accessible from the Render service.
- **Memory Limits**: The backend uses `sentence-transformers`, which can be memory-intensive. If the service crashes on the Free tier, consider upgrading to a larger instance on Render.
