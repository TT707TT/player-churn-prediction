import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import roc_auc_score, average_precision_score

st.set_page_config(page_title="Churn Risk Dashboard", layout="wide")

@st.cache_data
def load_data():
    preds = pd.read_csv("artifacts/test_predictions_gb.csv")
    with open("artifacts/gb_threshold_metrics.json", "r") as f:
        metrics = json.load(f)
    return preds, metrics

@st.cache_resource
def load_model():
    return joblib.load("artifacts/gb_churn_pipeline.joblib")

preds, metrics = load_data()
model = load_model()

st.title("Player Churn Risk Dashboard (Engagement Proxy)")
st.caption("Model: Gradient Boosting pipeline | Target: EngagementLevel = Low (proxy)")

# Sidebar controls
st.sidebar.header("Controls")
default_t = float(metrics["chosen_threshold"])
threshold = st.sidebar.slider("Risk threshold", 0.0, 1.0, float(default_t), 0.01)

# Metrics at chosen threshold using the stored test probabilities
y_true = preds["y_true"].to_numpy()
proba  = preds["proba_gb"].to_numpy()
y_pred = (proba >= threshold).astype(int)

def safe_div(a, b):
    return float(a / b) if b != 0 else 0.0

tp = int(((y_true == 1) & (y_pred == 1)).sum())
tn = int(((y_true == 0) & (y_pred == 0)).sum())
fp = int(((y_true == 0) & (y_pred == 1)).sum())
fn = int(((y_true == 1) & (y_pred == 0)).sum())

precision = safe_div(tp, tp + fp)
recall    = safe_div(tp, tp + fn)
f1        = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Threshold", f"{threshold:.2f}")
col2.metric("Precision", f"{precision:.3f}")
col3.metric("Recall", f"{recall:.3f}")
col4.metric("F1", f"{f1:.3f}")

st.subheader("Confusion Matrix (Test Set)")
cm_df = pd.DataFrame(
    [[tn, fp], [fn, tp]],
    index=["Actual 0", "Actual 1"],
    columns=["Pred 0", "Pred 1"]
)
st.dataframe(
    cm_df.style.background_gradient(cmap="Blues", axis=None).format("{:,}"),
    use_container_width=True
)

# Threshold comparison (t=0.50 vs chosen threshold), plus ROC-AUC / PR-AUC
with st.expander("Threshold comparison: t=0.50 vs chosen threshold"):
    roc_auc = roc_auc_score(y_true, proba)
    pr_auc  = average_precision_score(y_true, proba)

    cA, cB = st.columns(2)
    cA.metric("ROC-AUC", f"{roc_auc:.3f}")
    cB.metric("PR-AUC", f"{pr_auc:.3f}")

    t50_key   = "metrics_test_at_t0.5"
    tbest_key = f"metrics_test_at_t{default_t:.2f}"
    if t50_key in metrics and tbest_key in metrics:
        comp_df = pd.DataFrame({
            "t=0.50":              metrics[t50_key],
            f"t={default_t:.2f}":  metrics[tbest_key],
        })
        # ROC-AUC / PR-AUC already shown above as metric cards — drop from table
        comp_df = comp_df.drop(index=["ROC-AUC", "PR-AUC"], errors="ignore")
        st.dataframe(comp_df, use_container_width=True)

st.subheader("Risk Explorer")

view_mode = st.radio(
    "View",
    ["Top N by predicted risk", "Borderline cases (near threshold)"],
    horizontal=True
)

table = preds.copy()
table["pred_at_threshold"] = y_pred

if view_mode == "Top N by predicted risk":
    st.write("Top players by predicted churn risk (test set)")
    topn = st.slider("Show top N", 10, 200, 50, 10)
    table = table.sort_values("proba_gb", ascending=False).head(topn)
else:
    st.write("Players whose predicted risk is closest to the current threshold — "
             "these are the cases most sensitive to the threshold choice.")
    band = st.slider("Band width around threshold (±)", 0.01, 0.20, 0.05, 0.01)
    lower, upper = threshold - band, threshold + band
    table = table[(table["proba_gb"] >= lower) & (table["proba_gb"] <= upper)]
    table = table.sort_values("proba_gb", ascending=False)
    st.write(f"{len(table)} players with predicted risk in [{lower:.2f}, {upper:.2f}]")

priority_cols = ["Age", "Gender", "Location", "GameGenre", "GameDifficulty",
                  "proba_gb", "pred_at_threshold"]
priority_cols = [c for c in priority_cols if c in table.columns]
other_cols    = [c for c in table.columns if c not in priority_cols]

selected_cols = st.multiselect(
    "Columns to display",
    options=priority_cols + other_cols,
    default=priority_cols
)

display_table = table[selected_cols] if selected_cols else table

def highlight_flagged(row):
    if row.get("pred_at_threshold", 0) == 1:
        return ["background-color: #ffe5e5"] * len(row)
    return [""] * len(row)

if "pred_at_threshold" in display_table.columns:
    st.dataframe(
        display_table.style.apply(highlight_flagged, axis=1),
        use_container_width=True
    )
else:
    st.dataframe(display_table, use_container_width=True)

st.download_button(
    label="Download table as CSV",
    data=display_table.to_csv(index=True).encode("utf-8"),
    file_name="risk_explorer_export.csv",
    mime="text/csv"
)

st.subheader("What-if Prediction (Single Player)")
with st.expander("Enter a player profile to score"):
    c1, c2, c3 = st.columns(3)

    with c1:
        age      = st.number_input("Age", min_value=15, max_value=50, value=32)
        sessions = st.number_input("SessionsPerWeek", min_value=0, max_value=20, value=9)
        dur      = st.number_input("AvgSessionDurationMinutes", min_value=10, max_value=180, value=95)
        playtime = st.number_input("PlayTimeHours", min_value=0.0, max_value=24.0, value=12.0, step=0.1)

    with c2:
        level    = st.number_input("PlayerLevel", min_value=1, max_value=100, value=49)
        ach      = st.number_input("AchievementsUnlocked", min_value=0, max_value=50, value=25)
        purch    = st.selectbox("In-Game Purchases", [0, 1])
        gender   = st.selectbox("Gender", sorted(preds["Gender"].dropna().unique()) if "Gender" in preds.columns else ["Male", "Female"])

    with c3:
        location = st.selectbox("Location", sorted(preds["Location"].dropna().unique()) if "Location" in preds.columns else ["Asia", "USA", "EU"])
        genre    = st.selectbox("GameGenre", sorted(preds["GameGenre"].dropna().unique()) if "GameGenre" in preds.columns else ["Action", "RPG"])
        diff     = st.selectbox("GameDifficulty", sorted(preds["GameDifficulty"].dropna().unique()) if "GameDifficulty" in preds.columns else ["Easy", "Medium", "Hard"])

    row = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Location": location,
        "GameGenre": genre,
        "GameDifficulty": diff,
        "PlayTimeHours": playtime,
        "SessionsPerWeek": sessions,
        "AvgSessionDurationMinutes": dur,
        "PlayerLevel": level,
        "AchievementsUnlocked": ach,
        "InGamePurchases": purch
    }])

    score    = float(model.predict_proba(row)[:, 1][0])
    flagged  = score >= threshold

    rc1, rc2 = st.columns([1, 2])
    with rc1:
        st.metric("Predicted churn risk", f"{score:.3f}")
        if flagged:
            st.markdown(
                "<span style='background-color:#ffcccc;color:#a00;"
                "padding:4px 10px;border-radius:6px;font-weight:600;'>"
                "Flagged as churn-risk</span>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<span style='background-color:#d4f7d4;color:#1a7a1a;"
                "padding:4px 10px;border-radius:6px;font-weight:600;'>"
                "Not flagged</span>",
                unsafe_allow_html=True
            )
    with rc2:
        score_pct     = min(max(score, 0.0), 1.0) * 100
        threshold_pct = min(max(threshold, 0.0), 1.0) * 100
        bar_color = "#e57373" if flagged else "#64b5f6"
        st.markdown(
            f"""
            <div style="position:relative; width:100%; height:22px;
                        background-color:#eee; border-radius:4px; margin-top:8px;">
              <div style="position:absolute; left:0; top:0; height:100%;
                          width:{score_pct}%; background-color:{bar_color};
                          border-radius:4px;"></div>
              <div style="position:absolute; left:{threshold_pct}%; top:0;
                          height:100%; width:2px; background-color:#333;"></div>
              <div style="position:absolute; left:calc({threshold_pct}% - 18px);
                          top:24px; font-size:11px; color:#333;">
                t={threshold:.2f}
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )