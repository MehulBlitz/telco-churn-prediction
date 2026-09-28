# Publishing this project on GitHub

## 1. Get the code onto GitHub

### Option A — command line (from inside this project folder)

```bash
# one-time identity setup (skip if already configured)
git config --global user.name "Your Name"
git config --global user.email "you@example.com"

git init
git add .
git commit -m "Telco churn: pipelines, model comparison, Gradio app"
```

Create the empty repo on github.com first (**New repository** → name it
`telco-churn-prediction` → do *not* add a README), then:

```bash
git branch -M main
git remote add origin https://github.com/<your-username>/telco-churn-prediction.git
git push -u origin main
```

### Option B — GitHub Desktop (no terminal)

1. **File → Add local repository** → select this folder
   (it will say "not a repository" → click **Create a repository**).
2. Check the suggested `.gitignore` contents below are present, then
   **Publish repository**.

## 2. What gets committed (and what doesn't)

A `.gitignore` ships with this project. `data/` (the 240 KB CSV),
`artifacts/*.joblib` (the trained model) and any virtual environment are
excluded — everything else (code, README, figures, metrics) is included.

> **Tip:** a recruiter who clones the repo should run exactly two commands
> to see results — `pip install -r requirements.txt` then
> `python train_churn.py`. Keep it that way.

## 3. Optional 2-minute showcase: live demo on Hugging Face Spaces

1. Create a free account at https://huggingface.co → **New Space**.
2. SDK: **Gradio**, hardware: free CPU.
3. Upload: `app.py`, `requirements.txt`, `artifacts/churn_model.joblib`
   (small enough to commit for Spaces), and a `README.md` with YAML header:

```yaml
---
title: Telco Churn Predictor
emoji: 📉
sdk: gradio
app_file: app.py
---
```

4. The Space builds and hosts your interactive demo at a public URL you can
   put at the top of your GitHub README.

## 4. Repository housekeeping that looks professional

- **Description:** `Churn prediction on IBM Telco (7,043 rows) — sklearn
  pipelines, LogReg/RF/GB comparison, recall 55.9%→75.1% via threshold
  tuning, Gradio demo.`
- **Topics:** `machine-learning`, `scikit-learn`, `classification`,
  `churn-prediction`, `gradio`, `data-science-portfolio`
- Pin the repo on your profile; add the ROC curve and confusion matrix
  images to the README (they are already in `figures/`).
