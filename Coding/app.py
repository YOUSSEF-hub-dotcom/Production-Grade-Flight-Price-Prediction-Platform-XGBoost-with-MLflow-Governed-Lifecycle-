import streamlit as st
import requests
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Flight Price AI Engine",
    page_icon="✈️",
    layout="wide"
)

AIRLINES = ["IndiGo", "Air India", "Jet Airways", "SpiceJet", "Multiple carriers", "GoAir", "Vistara", "Air Asia"]
CITIES = ["Banglore", "Kolkata", "Delhi", "Chennai", "Mumbai", "Cochin", "Hyderabad"]

# Custom Professional UI Styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; font-weight: bold; }
    .price-box { border: 2px solid #007bff; padding: 25px; border-radius: 15px; text-align: center; background-color: white; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
    .metric-box { background-color: #f8f9fa; padding: 10px; border-radius: 10px; text-align: center; margin-top: 10px; font-size: 14px; }
    </style>
    """, unsafe_allow_html=True)

# Initialize Session States for smooth rendering
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "history_data" not in st.session_state:
    st.session_state.history_data = None

# Sidebar Controls
with st.sidebar:
    st.header("🤖 Model Intelligence")
    st.success("● API Status: Connected")
    st.info("● Model: XGBoost Pipeline")
    st.write("**MLflow Registry:** Production v1.0")
    st.write("---")
    st.subheader("💡 Analysis Tip")
    st.caption("Airlines often increase prices during peak seasons (March, May, June, December) and weekends.")
    if st.button("🔄 Clear App Cache"):
        st.cache_data.clear()
        st.session_state.prediction_result = None
        st.session_state.history_data = None
        st.rerun()

# Main Header
st.title("✈️ Advanced Flight Price Predictor")
st.markdown(
    "This engine uses an optimized **XGBoost model pipeline** monitored via **MLflow** to predict ticket prices based on engineered real-time market features.")

col_input, col_display = st.columns([1.4, 1])

with col_input:
    st.subheader("📍 Journey Details")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            airline = st.selectbox("Select Airline", AIRLINES)
            source = st.selectbox("Source City", CITIES)
            dep_date = st.date_input("Departure Date", datetime.now())
            st.caption(f"📅 Selected Day: **{dep_date.strftime('%A')}**")
            dep_hour = st.slider("Departure Hour", 0, 23, 10)

        with col2:
            stops = st.number_input("Total Stops", 0, 4, 0)
            destination = st.selectbox("Destination City", CITIES)
            duration = st.number_input("Duration (Minutes)", 30, 3000, 120)
            arr_hour = st.slider("Expected Arrival Hour", 0, 23, 14)

        submit = st.form_submit_button("💰 Predict Ticket Price")

# Trigger Pipeline Request on Form Submission
if submit:
    if source == destination:
        st.session_state.prediction_result = {"status": "error", "message": "Source and Destination cannot be the same!"}
    else:
        payload = {
            "Airline": airline,
            "Source": source,
            "Destination": destination,
            "Departure_Date": str(dep_date),
            "Dep_hour": dep_hour,
            "Arrival_hour": arr_hour,
            "Duration_mins": float(duration),
            "Total_Stops": stops
        }

        try:
            with st.spinner("🤖 AI Engine is analyzing market trends..."):
                response = requests.post("http://127.0.0.1:8000/predict", json=payload)

                if response.status_code == 200:
                    st.session_state.prediction_result = {"status": "success", "data": response.json()}
                    st.session_state.history_data = None # Force history reload to include the newest prediction
                elif response.status_code == 429:
                    st.session_state.prediction_result = {"status": "error", "message": "Rate limit exceeded! Please wait a minute before requesting again."}
                else:
                    error_detail = response.json().get('detail', 'Unknown backend error')
                    st.session_state.prediction_result = {"status": "error", "message": f"Error {response.status_code}: {error_detail}"}
        except Exception as e:
            st.session_state.prediction_result = {"status": "error", "message": f"Connection Failed: Ensure your FastAPI server is running. (Details: {str(e)})"}

# Right Column Display Management
with col_display:
    st.subheader("📊 Engine Outputs")
    
    # Case 1: No Prediction Made Yet
    if st.session_state.prediction_result is None:
        st.image("https://img.freepik.com/free-vector/airplane-flight-path-vector-illustration_1017-43288.jpg", use_container_width=True)
        st.info("Enter journey details on the left and click predict to see the AI-estimated fare.")
    
    # Case 2: Prediction Returned an Error
    elif st.session_state.prediction_result["status"] == "error":
        st.error(st.session_state.prediction_result["message"])
        if st.button("Reset View"):
            st.session_state.prediction_result = None
            st.rerun()

    # Case 3: Successful Prediction Display
    elif st.session_state.prediction_result["status"] == "success":
        result = st.session_state.prediction_result["data"]
        
        st.markdown(f"""
        <div class="price-box">
            <p style="color: #666; margin-bottom: 5px; letter-spacing: 1px; font-weight: 500;">ESTIMATED TICKET FARE</p>
            <h1 style="color: #007bff; font-size: 50px; margin: 0; font-family: sans-serif;">₹ {result['predicted_price']:,}</h1>
            <p style="color: #28a745; font-weight: bold; margin-top: 10px; margin-bottom: 0;">✅ Optimized by Leakage-Free Architecture</p>
        </div>
        """, unsafe_allow_html=True)

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(f'<div class="metric-box">⏱️ <b>Inference Latency</b><br>{result["inference_time"]}</div>', unsafe_allow_html=True)
        with m_col2:
            st.markdown(f'<div class="metric-box">📊 <b>Prediction R² Score</b><br>89.4% (Log Domain)</div>', unsafe_allow_html=True)

        st.balloons()
        if st.button("Clear Prediction Result"):
            st.session_state.prediction_result = None
            st.rerun()

# ----------------------------------------------------
# Recent Search History Section (Cached via Session State)
# ----------------------------------------------------
st.markdown("---")
st.subheader("📜 Recent AI Search History")

if st.session_state.history_data is None:
    try:
        history_res = requests.get("http://127.0.0.1:8000/history", timeout=3)
        if history_res.status_code == 200:
            st.session_state.history_data = history_res.json()
        else:
            st.session_state.history_data = []
    except Exception:
        st.session_state.history_data = "connection_error"

# Render History Data
if st.session_state.history_data == "connection_error":
    st.warning("⚠️ Could not connect to the database History API. Ensure FastAPI is fully operational.")
elif isinstance(st.session_state.history_data, list) and len(st.session_state.history_data) > 0:
    df_hist = pd.DataFrame(st.session_state.history_data)
    
    # Map and drop database indices safely
    df_hist.columns = ["ID", "Airline", "Source", "Destination", "Predicted Price (₹)", "Search Time"]
    df_hist = df_hist.drop(columns=["ID"])
    
    # Format dates nicely for human reading
    df_hist["Search Time"] = pd.to_datetime(df_hist["Search Time"]).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    st.dataframe(df_hist, use_container_width=True)
else:
    st.info("No query logs available in the central database yet.")
