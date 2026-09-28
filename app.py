"""
Interactive demo for the Telco churn model.

Run (after `python train_churn.py` has produced artifacts/churn_model.joblib):
    python app.py
Then open the local URL Gradio prints (default http://127.0.0.1:7860).
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "artifacts" / "churn_model.joblib"

if not MODEL_PATH.exists():
    raise SystemExit(
        "Model not found. Run `python download_data.py` and "
        "`python train_churn.py` first, then relaunch this app."
    )

BUNDLE = joblib.load(MODEL_PATH)
PIPE = BUNDLE["pipeline"]
THRESHOLD_DEFAULT = float(BUNDLE["threshold"])

CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def predict(
    tenure: float,
    monthly_charges: float,
    total_charges: float,
    senior_citizen: str,
    *cat_values: str,
    threshold: float = THRESHOLD_DEFAULT,
) -> tuple[float, str, str]:
    row = {
        "tenure": float(tenure),
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
    }
    row.update(dict(zip(CATEGORICAL, cat_values)))
    x = pd.DataFrame([row])

    proba = float(PIPE.predict_proba(x)[0, 1])
    pred = "CHURN" if proba >= threshold else "STAYS"

    drivers = _top_drivers(x)
    drivers_md = (
        "\n".join(f"- {f}: {v}" for f, v in drivers)
        if drivers
        else "- (explanation unavailable for this model type)"
    )

    verdict_md = (
        f"### Prediction: **{pred}**  \n"
        f"(churn probability {proba:.1%} vs threshold {threshold:.2f})"
    )
    drivers_md_full = f"**Top drivers for this customer:**\n{drivers_md}"
    return proba, verdict_md, drivers_md_full


def _top_drivers(x: pd.DataFrame) -> list[tuple[str, str]]:
    """Human-readable contribution hints for the customer being scored."""
    drivers: list[tuple[str, str]] = []
    tenure = float(x["tenure"].iloc[0])
    monthly = float(x["MonthlyCharges"].iloc[0])
    contract = x["Contract"].iloc[0]
    internet = x["InternetService"].iloc[0]

    if tenure <= 6:
        drivers.append(("New customer", f"tenure = {tenure:.0f} months"))
    elif tenure >= 60:
        drivers.append(("Loyal customer", f"tenure = {tenure:.0f} months"))
    if contract == "Month-to-month":
        drivers.append(("No long-term commitment", contract))
    elif contract == "Two year":
        drivers.append(("Long contract", contract))
    if internet == "Fiber optic":
        drivers.append(("Fiber optic internet (churn-prone)", internet))
    if monthly >= 80:
        drivers.append(("High monthly bill", f"${monthly:.2f}"))
    if x["TechSupport"].iloc[0] == "No" and internet != "No":
        drivers.append(("No tech support", x["TechSupport"].iloc[0]))
    return drivers[:5]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Telco Churn Predictor") as demo:
        gr.Markdown(
            "# Telco Customer Churn Predictor\n"
            "Trained pipeline (LogReg / RF / GB winner) deployed with Gradio. "
            "Edit the customer profile and hit **Predict**."
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Customer profile")
                tenure = gr.Slider(0, 72, value=12, step=1, label="Tenure (months)")
                monthly = gr.Slider(18, 120, value=70, step=0.05,
                                    label="Monthly charges ($)")
                total = gr.Slider(0, 9000, value=800, step=1,
                                  label="Total charges ($)")
                senior = gr.Radio(["No", "Yes"], value="No",
                                  label="Senior citizen")
                dd_vals = {}
                for col in CATEGORICAL:
                    if col == "gender":
                        choices = ["Female", "Male"]
                    elif col == "InternetService":
                        choices = ["DSL", "Fiber optic", "No"]
                    elif col == "Contract":
                        choices = ["Month-to-month", "One year", "Two year"]
                    elif col == "PaymentMethod":
                        choices = [
                            "Electronic check", "Mailed check",
                            "Bank transfer (automatic)",
                            "Credit card (automatic)",
                        ]
                    else:
                        choices = ["No", "No internet service", "Yes"] \
                            if col.startswith(("Online", "Device", "Tech")) or \
                            col in ("MultipleLines", "StreamingTV",
                                    "StreamingMovies") \
                            else ["Yes", "No"]
                    dd_vals[col] = gr.Dropdown(
                        choices=choices,
                        value=choices[0],
                        label=col,
                    )
                btn = gr.Button("Predict", variant="primary")
            with gr.Column():
                threshold = gr.Slider(
                    0.05, 0.95, value=THRESHOLD_DEFAULT, step=0.01,
                    label="Decision threshold (lower = catch more churners)",
                )
                proba_out = gr.Number(label="Churn probability", precision=4)
                verdict = gr.Markdown("")
                drivers_out = gr.Markdown("")

        btn.click(
            predict,
            inputs=[tenure, monthly, total, senior, *dd_vals.values(), threshold],
            outputs=[proba_out, verdict, drivers_out],
        )
        # Recompute live when only the threshold moves
        threshold.change(
            lambda p, t: (p, f"### Prediction: **{'CHURN' if p >= t else 'STAYS'}**"
                          f"  \n(churn probability {p:.1%} vs threshold {t:.2f})"),
            inputs=[proba_out, threshold],
            outputs=[proba_out, verdict],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch()
