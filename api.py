import logging
import time
from datetime import datetime
import jwt
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# ----------------------------------------------------
# Logging Configuration
# ----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("ticket_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TicketAPI")

# ----------------------------------------------------
# Database Setup (SQL Server)
# ----------------------------------------------------
DATABASE_URL = "mssql+pyodbc://./TicketDB?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PriceHistory(Base):
    __tablename__ = "price_history"
    id = Column(Integer, primary_key=True, index=True)
    airline = Column(String(100))
    source = Column(String(50))
    destination = Column(String(50))
    predicted_price = Column(Float)
    search_time = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ----------------------------------------------------
# Rate Limiting & Authentication Identifier
# ----------------------------------------------------
def get_smart_identifier(request: Request):
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        try:
            token = auth_header.split(" ")[1]
            payload = jwt.decode(token, options={"verify_signature": False})
            user_id = payload.get("sub")
            if user_id:
                return f"user:{user_id}"
        except Exception:
            pass
    return f"ip:{get_remote_address(request)}"

limiter = Limiter(key_func=get_smart_identifier)

# ----------------------------------------------------
# FastAPI App Initialization & Middleware
# ----------------------------------------------------
app = FastAPI(title="Advanced Ticket Price API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    allow_credentials=True,
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# ----------------------------------------------------
# MLflow Model Ingestion
# ----------------------------------------------------
MODEL_URI = "models:/TicketPricePredictor/Production"
try:
    logger.info(f"Loading Production Model from MLflow: {MODEL_URI}")
    model = mlflow.pyfunc.load_model(MODEL_URI)
    logger.info("Model loaded successfully into memory.")
except Exception as e:
    logger.critical(f"Failed to load model from MLflow Registry: {str(e)}")
    model = None

# ----------------------------------------------------
# Pydantic Schemas & Feature Adapters
# ----------------------------------------------------
class TicketRequestSimplified(BaseModel):
    Airline: str
    Source: str
    Destination: str
    Departure_Date: str = Field(..., description="Format: YYYY-MM-DD", json_schema_extra={"example": "2026-06-15"})
    Dep_hour: int = Field(..., ge=0, le=23)
    Arrival_hour: int = Field(..., ge=0, le=23)
    Duration_mins: float = Field(..., gt=0)
    Total_Stops: int = Field(..., ge=0, le=4)

def assign_session(hour):
    if 4 <= hour < 8: return 'Early Morning'
    elif 8 <= hour < 12: return 'Morning'
    elif 12 <= hour < 16: return 'Noon'
    elif 16 <= hour < 20: return 'Evening'
    elif 20 <= hour < 24: return 'Night'
    else: return 'Late Night'

def feature_adapter(payload: TicketRequestSimplified):
    date_obj = pd.to_datetime(payload.Departure_Date)

    # Re-engineered to exactly mirror core data pipeline rules
    full_data = {
        "Airline": str(payload.Airline),
        "Source": str(payload.Source),
        "Destination": str(payload.Destination),
        "Dep_Session": assign_session(payload.Dep_hour),
        "Total_Stops": float(payload.Total_Stops),
        "Dep_hour": int(payload.Dep_hour),
        "Arrival_hour": int(payload.Arrival_hour),
        "Duration_mins": int(payload.Duration_mins),
        "Month_of_Journey": int(date_obj.month),
        "Days_of_Journey": int(date_obj.day),
        "Day_of_Week": int(date_obj.dayofweek),
        "is_weekend": int(1 if date_obj.dayofweek >= 4 else 0),
        "Is_Long_Flight": int(1 if payload.Duration_mins > 480 else 0),
        "is_peak_season": int(1 if date_obj.month in [3, 5, 6, 12] else 0)
    }
    return full_data

# ----------------------------------------------------
# Database Dependencies & Helpers
# ----------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def log_to_db(data: dict, price: float):
    db = SessionLocal()
    try:
        new_entry = PriceHistory(
            airline=data['Airline'],
            source=data['Source'],
            destination=data['Destination'],
            predicted_price=price
        )
        db.add(new_entry)
        db.commit()
        logger.info(f"History Saved: {data['Airline']} | {data['Source']}->{data['Destination']} | Price: {price:.2f}")
    except Exception as e:
        db.rollback()
        logger.error(f"Database Logging Failed: {str(e)}", exc_info=True)
    finally:
        db.close()

# ----------------------------------------------------
# API Endpoints
# ----------------------------------------------------
@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, payload: TicketRequestSimplified, background_tasks: BackgroundTasks):
    if model is None:
        logger.critical("Model access attempted but model is not loaded!")
        raise HTTPException(status_code=503, detail="Model is currently unavailable in registry.")

    try:
        start_time = time.time()
        logger.info(f"Received inference request for Airline: {payload.Airline}")

        # 1. Transform input payload into raw features dictionary
        full_features = feature_adapter(payload)

        # 2. Convert to DataFrame
        input_df = pd.DataFrame([full_features])

        # 3. CRITICAL: Enforce exact Feature Column Order expected by MLflow wrapped Pipeline
        feature_order = [
            'Airline', 'Source', 'Destination', 'Dep_Session', 'Total_Stops', 
            'Dep_hour', 'Arrival_hour', 'Duration_mins', 'Month_of_Journey', 
            'Days_of_Journey', 'Day_of_Week', 'is_weekend', 'Is_Long_Flight', 'is_peak_season'
        ]
        input_df = input_df[feature_order]

        # 4. Enforce strict typing before pipeline ingestion
        input_df = input_df.astype({
            'Total_Stops': 'float64',
            'Dep_hour': 'int32',
            'Arrival_hour': 'int32',
            'Duration_mins': 'int64',
            'Month_of_Journey': 'int32',
            'Days_of_Journey': 'int32',
            'Day_of_Week': 'int32',
            'is_weekend': 'int64',
            'Is_Long_Flight': 'int64',
            'is_peak_season': 'int64'
        })

        # 5. Model Inference (Wrapper automatically handles scaling/encoding/expm1 conversion)
        prediction = model.predict(input_df)
        final_price = float(prediction[0])

        inf_time_ms = round((time.time() - start_time) * 1000, 2)
        inf_time_str = f"{inf_time_ms}ms"

        logger.info(f"Prediction successful. Price: {final_price:.2f} | Latency: {inf_time_str}")

        # 6. Offload logging to DB via Background Task
        background_tasks.add_task(log_to_db, full_features, final_price)

        return {
            "predicted_price": round(final_price, 2),
            "inference_time": inf_time_str,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error during prediction endpoint execution: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Data Transformation / Inference Error: {str(e)}")

@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    return db.query(PriceHistory).order_by(PriceHistory.search_time.desc()).limit(5).all()
