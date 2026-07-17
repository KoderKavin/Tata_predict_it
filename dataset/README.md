# Step 1 — Vayu Object Storage (Dataset)

**Predict-It** › **Vayu Object Storage** · `dataset/`

| | |
|---|---|
| **Previous** | [Step 0 — Vayu AI Studio Workspace](../00_vayu_workspaces/) |
| **Next** | [Step 2 — Vayu MLflow →](../02_vayu_mlflow/) |

This folder holds BBC-style news CSVs for training and batch inference. If `news.csv` and `validation.csv` are already here, you can skip S3 sync. Otherwise, upload or download them from **Vayu Object Storage (S3)** using the notebook below.

📄 **Reference docs:** [Vayu S3 Object Storage Documentation](https://ipcloud.tatacommunications.com/docs/docs/s3-object-browser/)

---

## Folder Contents

| File / Folder | Columns / purpose |
|---------------|-------------------|
| `news.csv` | `ArticleId`, `Text`, `Category` — labeled training (~1.5k rows) |
| `validation.csv` | `ArticleId`, `Text` — unlabeled submission set (~735 rows) |
| `00_dataset.ipynb` | Upload and download snippets for syncing CSVs with S3 (boto3) |

**Categories:** `business`, `entertainment`, `politics`, `sport`, `tech`.

---

## Quick Start

1. **Check locally:** Confirm `news.csv` and `validation.csv` are present. If they are, continue to [Step 2](../02_vayu_mlflow/).

2. **Pull from S3 (if needed):** Open [`00_dataset.ipynb`](00_dataset.ipynb), then select the notebook kernel:

   1. Open **Select Kernel** and choose **Python Environments**.

      ![Select Kernel — Python Environments](../assets/kernel_select.png)

   2. Under **Select a Python Environment**, pick the **Recommended** environment (it should point to the `.venv` from [Step 0](../00_vayu_workspaces/)).

      ![Select a Python Environment](../assets/Select_kernerl_env.png)

   3. **Validate the path:** Confirm the selected interpreter path ends with `<your-env-name>/bin/python` (for example, `.venv/bin/python` if you created `.venv` in [Step 0](../00_vayu_workspaces/)).

   Run the **Download Dataset from Vayu Cloud Storage** cell (credentials load from the root `.env` — see [overview](../README.md#minimal-run)).

3. **Upload to S3 (optional):** Use the **Upload Dataset to Vayu Cloud Storage** cell in the same notebook if you want a copy stored in your bucket.

> **Tip:** `validation.csv` has **no** `Category` column — use it only for final predictions, not tuning metrics.

`starter_kit/train.ipynb` reads:

```python
df_labeled = pd.read_csv("../dataset/news.csv")
df_inference = pd.read_csv("../dataset/validation.csv")
```

---

## Data notes

| File | Use |
|------|-----|
| `news.csv` | Training + 20% stratified holdout inside notebook |
| `validation.csv` | Batch predict only (no labels) |

---

## Navigation

| | |
|---|---|
| **Previous** | [Step 0 — Vayu AI Studio Workspace](../00_vayu_workspaces/) |
| **Next** | [Step 2 — Vayu MLflow →](../02_vayu_mlflow/) |
| **Overview** | [Predict-It overview](../README.md) |
