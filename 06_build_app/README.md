# Step 6 — Build App (Streamlit UI)

**Predict-It** › **Streamlit UI & Docker** · `06_build_app/`

| | |
|---|---|
| **⬅ Previous** | [Step 5 — Deploy Model](../05_deploy_model/) |
| **Next ➡** | [Step 7 — Deploy ML Service →](../07_deploy/) |
| **🏠 Overview** | [Predict-It overview](../README.md) |

Run the **Predict-It Streamlit UI** locally, then continue to [Step 7](../07_deploy/) to build, sign, and deploy the Docker image as a **Vayu ML Service**.

The UI is a thin client: it POSTs article text to your **Vayu Model Serving** endpoint from [Step 5](../05_deploy_model/) and displays predicted categories.

---

## Folder Contents

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI — **General** and **Live stream** tabs |
| `Dockerfile` | Container image for Vayu ML Service |
| `.dockerignore` | Keeps local junk out of the image |

---

## Prerequisites

| Step | Folder | You need |
|------|--------|----------|
| 5 | [`05_deploy_model/`](../05_deploy_model/) | Predict URL / host + model name from Model Serving |

Configure in your root `.env` (see [overview](../README.md#minimal-run)):

```bash
VAYU_PREDICT_HOST=https://<>.cloudservices.tatacommunications.com
VAYU_MODEL_NAME=<model-name>
VAYU_HOST_HEADER=<hostname-from-deployment-status>
```

Use the Model Serving **base URL** only from [Step 5](../05_deploy_model/) — no `/v1/models/...:predict` suffix.

---

## Quick Start

### 1. Run locally

```bash
cd 06_build_app
streamlit run app.py
```

`app.py` loads `predict-it/.env` automatically via `load_dotenv`. In the **sidebar**, confirm **Host** and **Model name** match your deployment (defaults come from `VAYU_PREDICT_HOST` / `VAYU_MODEL_NAME`).

- **General** — paste a single article → category prediction
- **Live stream** — feeds `01_dataset/news.csv` with labels from `category_labels.json`

When local testing works, continue to [Step 7](../07_deploy/) to build, sign, and push the Docker image.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Prediction errors | Confirm Model Serving is **Ready** and the predict URL from [Step 5](../05_deploy_model/) is correct |
| Host / 403 errors | Set `VAYU_HOST_HEADER` in `.env` or the Streamlit sidebar |
| UI shows raw codes (`0`, `1`) | Check `category_labels.json` was generated during training in [Step 3](../03_starter_kit/) |
| Live stream empty | Confirm `01_dataset/news.csv` exists (included in the Docker image) |

---

## Pro tips

- Never `COPY` your `.env` into the Docker image — pass credentials via the ML Service wizard in [Step 7](../07_deploy/).
- Keep `VAYU_PREDICT_HOST`, `VAYU_MODEL_NAME`, and the predict URL path in sync.

---

## Navigation

| | |
|---|---|
| **⬅ Previous** | [Step 5 — Deploy Model](../05_deploy_model/) |
| **Next ➡** | [Step 7 — Deploy ML Service →](../07_deploy/) |
| **🏠 Overview** | [Predict-It overview](../README.md) |
