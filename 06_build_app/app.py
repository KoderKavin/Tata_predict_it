"""
Predict It — Streamlit UI for Dynamic Airline Pricing.

streamlit run app.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import joblib
from dotenv import load_dotenv, find_dotenv

# Automatically find .env in the directory tree
load_dotenv(find_dotenv())
# Use PROJECT_ROOT from .env if set, otherwise fallback to the directory above this script
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))

# ── paths ──────────────────────────────────────────────────────────────────────
FLIGHT_CSV = ROOT / "01_dataset" / "flight_data.csv"
LOCAL_MODEL_PATH = ROOT / "03_starter_kit" / "model.joblib"

@st.cache_resource
def load_local_model():
    if not LOCAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Local model not found at {LOCAL_MODEL_PATH}. Please run train.py first.")
    return joblib.load(LOCAL_MODEL_PATH)


def predict_fare(
    features: list[list[float | int]]
) -> list[float]:
    if not features:
        return []

    # ── Offline / Local Inference ──
    model = load_local_model()
    cols = ['DaysToDeparture', 'BookedSeats', 'TotalCapacity', 'CompetitorPrice', 'SeasonalityIndex']
    df_features = pd.DataFrame(features, columns=cols)
    preds = model.predict(df_features).tolist()
    return preds


# ── UI helpers ─────────────────────────────────────────────────────────────────
def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        .block-container { 
            padding-top: 3rem; 
            padding-bottom: 3rem; 
            max-width: 1100px; 
        }
        h1.hero, .hero { 
            font-size: 4.5rem !important; 
            font-weight: 800 !important; 
            letter-spacing: -0.04em !important; 
            margin-top: 0px !important;
            margin-bottom: 0.5rem !important;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            animation: gradient-shift 5s ease infinite !important;
            background-size: 200% 200% !important;
            text-align: center !important;
            line-height: 1.2 !important;
            padding: 0.1em 0 !important;
            display: block !important;
        }
        @keyframes gradient-shift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        p.subhero, .subhero { 
            color: #64748b !important; 
            font-size: 1.35rem !important; 
            margin-bottom: 2.5rem !important; 
            text-align: center !important;
            display: block !important;
        }
        .pred-card {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(226, 232, 240, 0.8);
            border-radius: 16px;
            padding: 1.5rem 1.75rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .pred-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.08), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            border-color: rgba(139, 92, 246, 0.3);
        }
        .pred-meta { 
            font-size: 0.8rem; 
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8; 
            margin-bottom: 0.5rem; 
        }
        .pred-snippet { 
            font-size: 0.95rem; 
            line-height: 1.6; 
            color: #334155; 
            margin-bottom: 1.25rem;
        }
        .fare-badge {
            display: inline-block;
            padding: 0.4rem 1rem;
            border-radius: 9999px;
            font-size: 1.15rem;
            font-weight: 800;
            color: white;
            letter-spacing: 0.02em;
            background: linear-gradient(135deg, #10b981, #059669);
            box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.3);
        }
        .stream-counter {
            font-size: 0.9rem;
            color: #64748b;
            padding: 0.5rem 0;
        }
        
        /* Hide sidebar and modern buttons */
        [data-testid="collapsedControl"] { display: none; }
        section[data-testid="stSidebar"] { display: none; }
        
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s;
        }
        .stButton>button:hover {
            transform: scale(1.02);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_card(
    flight_id: Any,
    details: dict[str, Any],
    predicted_fare: float,
    *,
    show_truth: bool = False,
    true_fare: float | None = None,
) -> None:
    details_str = " | ".join(f"{k}: {v}" for k, v in details.items())
    truth_html = ""
    if show_truth and true_fare is not None:
        truth_html = f' &nbsp;·&nbsp; optimal base: <strong>${true_fare:.2f}</strong>'
        
    st.markdown(
        f"""
        <div class="pred-card">
            <div class="pred-meta">Flight ID: {flight_id}{truth_html}</div>
            <div class="pred-snippet">{details_str}</div>
            <div style="margin-top:0.6rem">
                <span class="fare-badge">Recommended Fare: ${predicted_fare:.2f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )





# ── pages ──────────────────────────────────────────────────────────────────────
def page_single_predict() -> None:
    st.markdown("#### Scenario Planner")
    st.write("Input flight scenario details to get a dynamic fare recommendation.")
    
    col1, col2 = st.columns(2)
    with col1:
        days_to_departure = st.number_input("Days to Departure", min_value=1, max_value=365, value=30)
        total_capacity = st.number_input("Total Capacity", min_value=10, max_value=800, value=180)
        seasonality = st.slider("Seasonality Index", 0.5, 2.0, 1.0, 0.1)
    with col2:
        booked_seats = st.number_input("Booked Seats", min_value=0, max_value=800, value=90)
        competitor_price = st.number_input("Competitor Price ($)", min_value=10.0, max_value=5000.0, value=150.0)

    if st.button("Get Recommended Fare", type="primary"):
        if booked_seats > total_capacity:
            st.error("Booked seats cannot exceed total capacity.")
            return

        feature_vector = [[days_to_departure, booked_seats, total_capacity, competitor_price, seasonality]]
        
        with st.spinner("Calculating optimal fare…"):
            try:
                preds = predict_fare(feature_vector)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return
                
        pred_fare = preds[0]
        details = {
            "Days Out": days_to_departure,
            "Booked": f"{booked_seats}/{total_capacity}",
            "Competitor": f"${competitor_price:.2f}",
            "Seasonality": seasonality
        }
        render_prediction_card("Scenario-01", details, pred_fare)


def page_live_stream() -> None:
    st.markdown("#### Live Pricing Feed")
    st.caption("Streams flight scenarios from a synthetic dataset and applies dynamic pricing.")

    if not FLIGHT_CSV.is_file():
        st.warning(f"Dataset not found: {FLIGHT_CSV}. Please run the training script to generate it.")
        return

    df = pd.read_csv(FLIGHT_CSV)
    has_truth = "OptimalFare" in df.columns

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        delay = st.slider("Delay between requests (seconds)", 0.3, 5.0, 1.0, 0.1)
    with c2:
        max_flights = st.number_input(
            "Max flights",
            min_value=1,
            max_value=len(df),
            value=min(50, len(df)),
            step=1,
        )
    with c3:
        shuffle = st.checkbox("Shuffle", value=False)

    subset = df.sample(n=max_flights, random_state=42) if shuffle else df.head(max_flights)

    col_start, col_stop, _ = st.columns([1, 1, 4])
    with col_start:
        start = st.button("▶ Start stream", type="primary")
    with col_stop:
        stop = st.button("■ Stop")

    if stop:
        st.session_state.stream_running = False

    status_slot = st.empty()
    feed_slot = st.container()
    history: list[dict[str, Any]] = []

    if start:
        st.session_state.stream_running = True

    if st.session_state.get("stream_running"):
        progress = st.progress(0.0)
        for i, (_, row) in enumerate(subset.iterrows()):
            if not st.session_state.get("stream_running", False):
                status_slot.info("Stream stopped.")
                break

            flight_id = row.get("FlightId", f"FL-{i+1}")
            
            features = [
                row["DaysToDeparture"],
                row["BookedSeats"],
                row["TotalCapacity"],
                row["CompetitorPrice"],
                row["SeasonalityIndex"]
            ]
            
            true_fare = float(row["OptimalFare"]) if has_truth else None

            status_slot.markdown(
                f'<div class="stream-counter">Processing flight {i + 1} of {len(subset)}…</div>',
                unsafe_allow_html=True,
            )

            try:
                preds = predict_fare([features])
                pred_fare = preds[0]
            except Exception as exc:
                st.session_state.stream_running = False
                status_slot.error(f"Prediction failed at flight {i + 1}: {exc}")
                break

            details = {
                "Days Out": row["DaysToDeparture"],
                "Booked": f"{row['BookedSeats']}/{row['TotalCapacity']}",
                "Competitor": f"${row['CompetitorPrice']:.2f}",
                "Seasonality": row["SeasonalityIndex"]
            }

            entry = {
                "FlightId": flight_id,
                "Details": details,
                "PredictedFare": pred_fare,
                "TrueFare": true_fare,
            }
            history.insert(0, entry)

            with feed_slot:
                st.markdown(
                    f'<div class="stream-counter">Latest predictions ({len(history)} shown)</div>',
                    unsafe_allow_html=True,
                )
                for item in history[:20]:
                    render_prediction_card(
                        item["FlightId"],
                        item["Details"],
                        item["PredictedFare"],
                        show_truth=has_truth,
                        true_fare=item["TrueFare"],
                    )

            progress.progress((i + 1) / len(subset))
            time.sleep(delay)

        else:
            st.session_state.stream_running = False
            status_slot.success(f"Finished — priced {len(subset)} flights.")


def main() -> None:
    st.set_page_config(page_title="Airline Yield Manager", page_icon="✈️", layout="wide")
    inject_styles()

    if "stream_running" not in st.session_state:
        st.session_state.stream_running = False

    st.markdown('<h1 class="hero">Airline Yield Manager</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subhero">Dynamic pricing intelligence.</p>',
        unsafe_allow_html=True,
    )


    tab_single, tab_stream = st.tabs(["Scenario Planner", "Live Feed"])

    with tab_single:
        page_single_predict()

    with tab_stream:
        page_live_stream()


if __name__ == "__main__":
    main()
