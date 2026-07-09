# ✈️ Flight Ticket Price Prediction
## Production-Ready Machine Learning Platform for Intelligent Airline Ticket Price Prediction

> **An end-to-end Machine Learning & MLOps solution that predicts airline ticket prices using advanced feature engineering, XGBoost, MLflow, FastAPI, and Streamlit.**
>
> Designed with a **Data Scientist / ML Engineer mindset**, focusing on reproducibility, scalability, explainability, and real-world business value.

---

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-success?style=for-the-badge)
![MLflow](https://img.shields.io/badge/MLflow-MLOps-blue?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-orange?style=for-the-badge)

---

# 🌟 Project Overview

Predicting airline ticket prices is one of the most challenging regression problems due to the dynamic nature of airline pricing.

Ticket prices fluctuate continuously based on numerous business factors, including:

- Airline competition
- Route popularity
- Travel season
- Departure time
- Flight duration
- Number of stops
- Customer demand

This project builds an intelligent Machine Learning platform capable of estimating ticket prices while also extracting valuable business insights that can support pricing strategies and revenue optimization.

Unlike traditional ML projects, this repository demonstrates the **complete production lifecycle**, starting from raw data all the way to deployment using modern MLOps practices.

---

# 🚀 Why This Project?

Most machine learning repositories stop after training a model.

This project goes much further by covering the complete lifecycle:

```
Business Understanding
        ↓
Data Cleaning
        ↓
EDA
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Hyperparameter Optimization
        ↓
MLflow Experiment Tracking
        ↓
Model Registry
        ↓
Production Quality Gate
        ↓
FastAPI Deployment
        ↓
Streamlit Dashboard
```

The objective is not only to build an accurate prediction model, but also to demonstrate how a real-world Machine Learning solution should be developed, monitored, versioned, and deployed.

---

# 📚 Table of Contents

- Project Overview
- Business Problem
- Business Objectives
- Key Features
- Technology Stack
- Project Architecture
- Dataset Overview
- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis
- Business Insights
- Model Development
- Hyperparameter Optimization
- Model Evaluation
- MLflow Lifecycle
- FastAPI Deployment
- Streamlit Dashboard
- Installation
- Quick Start
- API Documentation
- Project Structure
- Future Improvements
- Documentation
- License
- Author

---

# 💼 Business Problem

Airline ticket pricing is influenced by a complex interaction between demand, seasonality, competition, operational costs, and customer behavior.

Traditional pricing strategies often rely on historical averages or manual adjustments, making it difficult to react quickly to changing market conditions.

This project aims to solve that challenge by providing a predictive model capable of estimating ticket prices while uncovering hidden pricing patterns.

The generated insights can assist multiple stakeholders:

### ✈️ Airlines

- Dynamic pricing strategies
- Revenue optimization
- Route planning
- Demand forecasting
- Competitive pricing

### 👨‍💼 Revenue Managers

- Better pricing decisions
- Market segmentation
- Seasonal planning
- Capacity optimization

### 🧳 Travelers

- Better booking decisions
- Travel budget estimation
- Price comparison
- Cost-saving opportunities

---

# 🎯 Project Objectives

The primary objectives of this project are:

✅ Build a highly accurate regression model

✅ Engineer meaningful business-oriented features

✅ Perform comprehensive exploratory data analysis

✅ Identify the most influential pricing factors

✅ Build a reproducible ML pipeline

✅ Track every experiment using MLflow

✅ Register production-ready models

✅ Deploy prediction services through FastAPI

✅ Build an interactive Streamlit dashboard

✅ Generate actionable business recommendations

---

# ⭐ Key Features

### 🤖 Machine Learning

- XGBoost Regressor
- Advanced Feature Engineering
- Hyperparameter Optimization
- Cross Validation
- Sample Weighting
- Log Transformation
- Outlier Removal

---

### 📊 Data Analytics

- Business-driven EDA
- Statistical Analysis
- Correlation Analysis
- Trend Analysis
- Seasonal Analysis
- Route Analysis
- Airline Comparison

---

### ⚙️ MLOps

- MLflow Tracking
- MLflow Model Registry
- MLflow Projects
- Model Versioning
- Model Signature
- Automated Quality Gates
- Reproducible Training Pipeline

---

### 🚀 Deployment

- FastAPI REST API
- Streamlit Dashboard
- Production-ready Prediction Service
- Interactive User Interface

---

### 📈 Business Analytics

- Revenue Optimization Insights
- Pricing Strategy Recommendations
- Route Analysis
- Airline Benchmarking
- Seasonal Pricing Analysis
- Customer Decision Support

---

# 🛠 Technology Stack

| Category | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Machine Learning | XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Hyperparameter Tuning | GridSearchCV, RandomizedSearchCV |
| Experiment Tracking | MLflow |
| Model Registry | MLflow Registry |
| API Framework | FastAPI |
| Dashboard | Streamlit |
| Environment | Conda |
| Version Control | Git & GitHub |

---

# 🏗 Project Architecture

```text
                        Flight Dataset
                              │
                              ▼
                  Data Cleaning & Validation
                              │
                              ▼
                   Feature Engineering Pipeline
                              │
                              ▼
                Exploratory Data Analysis (EDA)
                              │
                              ▼
                   XGBoost Model Training
                              │
                              ▼
               Hyperparameter Optimization
                              │
                              ▼
                 Performance Evaluation
                              │
                              ▼
                  MLflow Experiment Tracking
                              │
                              ▼
                   Model Registry & Versioning
                              │
                              ▼
                 Production Quality Gate
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
      FastAPI REST API                 Streamlit Dashboard
             │                                 │
             └──────────────┬──────────────────┘
                            ▼
                     End Users / Business
```

---

> 📌 **Next Section:** Dataset Overview → Data Cleaning → Feature Engineering → Exploratory Data Analysis → Business Insights

---

# 📊 Dataset Overview

The project is built on a real-world airline ticket pricing dataset collected from multiple airlines operating across major Indian cities.

Unlike synthetic datasets often used in academic projects, this dataset captures realistic pricing behavior affected by airline competition, travel demand, route popularity, seasonality, and operational constraints.

### Dataset Summary

| Property | Value |
|-----------|-------|
| Domain | Aviation & Revenue Management |
| Dataset Format | Excel (.xlsx) |
| Problem Type | Supervised Regression |
| Target Variable | **Price** |
| Samples | Airline Ticket Records |
| Missing Values | Present (Handled) |
| Duplicates | Removed |
| Outliers | Detected & Treated |

---

# 🎯 Prediction Target

The objective is to estimate the final airline ticket price using only information available before booking.

Target Variable

```text
Price
```

Unlike a classification problem, price prediction requires the model to learn complex nonlinear relationships between multiple categorical and numerical variables.

---

# 📋 Features Used

The model leverages multiple business-related variables that directly influence airline pricing.

| Feature | Description |
|----------|-------------|
| Airline | Airline company |
| Source | Departure city |
| Destination | Arrival city |
| Total Stops | Number of flight stops |
| Duration | Flight duration |
| Journey Date | Travel date |
| Departure Time | Flight departure |
| Arrival Time | Flight arrival |

These variables collectively represent customer demand, operational costs, and route competitiveness.

---

# 🧹 Data Cleaning & Preprocessing

Real-world datasets are rarely clean.

A robust preprocessing pipeline was developed to improve model reliability and ensure reproducibility.

### Missing Value Handling

Missing values were analyzed individually.

Instead of removing valuable observations, appropriate imputation techniques were applied.

✔ Mode Imputation

---

### Duplicate Removal

Duplicate observations were removed to eliminate potential data leakage.

Benefits:

- Improved model generalization
- Reduced overfitting
- Better evaluation reliability

---

### Date Processing

The original journey date was transformed into meaningful business features.

Extracted Features

- Journey Day
- Journey Month
- Day of Week
- Quarter

These variables allow the model to capture seasonal pricing patterns.

---

### Time Processing

Departure and arrival timestamps were decomposed into:

- Hour
- Minute

These features capture customer demand variations throughout the day.

---

### Flight Duration Engineering

Flight duration originally existed as text.

Example

```text
2h 50m
```

Converted into

```text
170 Minutes
```

This transformation allows regression algorithms to better understand operational cost differences.

---

### Stops Encoding

Categorical stop information was converted into numerical values.

Example

| Original | Encoded |
|-----------|----------|
| Non-stop | 0 |
| 1 stop | 1 |
| 2 stops | 2 |
| 3 stops | 3 |

---

### Target Transformation

Ticket prices exhibited strong right-skewness.

To stabilize variance and reduce extreme values, a logarithmic transformation was applied.

```python
Price = log1p(Price)
```

Benefits

- Better optimization
- Reduced impact of expensive flights
- Improved prediction stability

---

### Outlier Detection

Price outliers were identified using the IQR method.

Rather than allowing extreme prices to dominate the learning process, unrealistic observations were removed.

Benefits

✔ Better generalization

✔ More stable model

✔ Lower prediction error

---

# ⚙️ Feature Engineering

Feature Engineering played a major role in improving prediction performance.

Instead of relying solely on raw variables, several business-oriented features were introduced.

---

## ✈️ Dep_Session

Departure time was categorized into meaningful time periods.

Examples

- Early Morning
- Morning
- Afternoon
- Evening
- Night

Business Motivation

Customer demand differs significantly throughout the day.

Morning and evening flights often carry premium prices due to business travel demand.

---

## 🌍 Route Feature

Created by combining

```
Source + Destination
```

Example

```
Delhi → Cochin
```

Instead of treating cities independently, this captures complete route economics.

---

## ⏱ Long Flight Indicator

A binary feature identifying long-haul flights.

```text
Duration > Threshold
```

Business Motivation

Long flights generally incur:

- Higher fuel consumption
- More crew expenses
- Higher operational costs

---

## 📅 Peak Season Indicator

Created using travel month.

Allows the model to identify:

- Holiday periods
- High-demand seasons
- Vacation months

---

## 🎯 Weekend Indicator

Binary variable

Weekend

Weekday

Useful because customer travel behavior changes significantly during weekends.

---

# 📊 Exploratory Data Analysis (EDA)

The purpose of EDA was not simply visualization.

Instead, it focused on discovering actionable business insights capable of improving pricing strategies.

The analysis answered questions such as:

- Which airlines charge premium prices?
- How much do stops influence pricing?
- Which cities are the most expensive?
- How does seasonality affect demand?
- Are morning flights consistently more expensive?
- Does duration significantly influence prices?

---

# 📈 Major Business Findings

## ✈️ Airline Pricing Strategy

Premium airlines consistently charge higher prices than budget carriers.

Key observations:

- Premium airlines achieve higher average revenue per ticket.
- Budget airlines compete through lower prices and higher passenger volume.
- Some carriers exhibit high pricing volatility, suggesting opportunities for pricing optimization.

Business Impact

- Revenue Optimization
- Customer Segmentation
- Competitive Positioning

---

## 🌍 Geographic Pricing Dynamics

Route selection has a significant impact on ticket prices.

Major metropolitan routes generally command premium pricing due to stronger business demand.

Business Opportunities

- Dynamic route pricing
- Capacity planning
- Hub optimization

---

## ⏱ Flight Duration Analysis

Long-haul flights demonstrate a clear pricing premium.

Operational factors contributing to higher prices include:

- Fuel consumption
- Crew costs
- Aircraft utilization
- Passenger services

---

## 📅 Seasonal Pricing Patterns

Ticket prices vary considerably throughout the year.

Peak travel seasons consistently produce higher prices than off-season periods.

Business Benefits

- Better demand forecasting
- Seasonal revenue planning
- Promotional campaign optimization

---

## 🕐 Time-of-Day Analysis

Departure time significantly influences pricing.

Morning and evening departures generally command premium prices because of higher business demand.

Late-night flights tend to offer lower fares, creating opportunities for price-sensitive travelers.

---

# 💡 Business Insights Summary

The EDA produced several actionable findings that extend beyond model performance.

Key insights include:

- Premium airlines maintain higher margins through service differentiation.
- Budget carriers compete on volume rather than price.
- Long-haul routes justify premium pricing due to operational costs.
- High-demand routes generate greater revenue opportunities.
- Seasonal demand creates significant pricing fluctuations.
- Travel time strongly affects customer willingness to pay.
- Route-specific pricing strategies outperform generic pricing models.
- Machine learning can support dynamic pricing decisions using historical flight characteristics.

---

> 📌 **Next Section:** Model Development → Hyperparameter Optimization → Model Evaluation → MLflow Lifecycle → Production Deployment

---

# 🤖 Model Development

Selecting the right machine learning algorithm is one of the most critical decisions in any predictive analytics project.

Rather than choosing the most complex model, the objective was to identify a solution that delivers an optimal balance between prediction accuracy, computational efficiency, interpretability, and production readiness.

Several machine learning algorithms were evaluated throughout the experimentation phase before selecting the final production model.

---

# 🎯 Why XGBoost?

Among all evaluated models, **Extreme Gradient Boosting (XGBoost)** consistently achieved the best overall performance.

The selection was based on both quantitative evaluation metrics and qualitative production considerations.

### Advantages of XGBoost

✅ Excellent performance on structured/tabular datasets

✅ Robust handling of nonlinear feature interactions

✅ Built-in regularization to reduce overfitting

✅ High computational efficiency

✅ Strong scalability for large datasets

✅ Feature importance interpretation

✅ Proven reliability in production environments

Unlike linear models, XGBoost captures complex relationships between flight characteristics and ticket prices, making it particularly suitable for airline pricing prediction.

---

# 🔬 Model Selection Process

Multiple regression algorithms were explored during experimentation.

| Model | Purpose |
|---------|---------|
| Linear Regression | Baseline Model |
| Decision Tree | Nonlinear Benchmark |
| Random Forest | Ensemble Comparison |
| Gradient Boosting | Boosting Baseline |
| **XGBoost** | ✅ Final Production Model |

The comparison considered:

- Prediction Accuracy
- Generalization Performance
- Training Time
- Inference Speed
- Robustness Against Overfitting
- Production Suitability

After extensive evaluation, **XGBoost** demonstrated the best balance across all criteria.

---

# ⚙️ Training Pipeline

The complete training workflow was designed to ensure reproducibility and maintainability.

```text
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Data Encoding
      │
      ▼
Train / Validation Split
      │
      ▼
Hyperparameter Optimization
      │
      ▼
Model Training
      │
      ▼
Performance Evaluation
      │
      ▼
MLflow Tracking
      │
      ▼
Model Registry
      │
      ▼
Production Deployment
```

Every experiment followed the same standardized pipeline to guarantee reproducible results.

---

# ⚙️ Hyperparameter Optimization

Model performance was further enhanced through systematic hyperparameter optimization.

Instead of relying on default configurations, multiple search strategies were employed to identify the optimal parameter combination.

### Optimization Techniques

- Randomized Search CV
- Grid Search CV
- Cross Validation

The optimization focused on parameters such as:

- Number of Trees
- Maximum Tree Depth
- Learning Rate
- Minimum Child Weight
- Gamma
- Subsample Ratio
- Column Sampling Ratio

This process significantly improved model generalization while reducing overfitting.

---

# 📊 Cross Validation Strategy

To ensure reliable performance estimation, **K-Fold Cross Validation** was adopted during model development.

Benefits include:

- Reduced evaluation bias
- Better generalization estimation
- Stable performance measurement
- Robust model selection

Cross-validation results demonstrated consistent performance across multiple folds, indicating strong model stability.

---

# 📈 Model Performance

The final production model achieved strong predictive performance across multiple regression metrics.

| Metric | Score |
|---------|-------:|
| R² Score | **0.86+** |
| Mean Absolute Error (MAE) | Low |
| Root Mean Squared Error (RMSE) | Low |
| Mean Absolute Percentage Error (MAPE) | Low |
| Cross Validation Stability | High |

These results confirm that the model captures the underlying pricing patterns effectively while maintaining good generalization.

---

# 📉 Error Analysis

Beyond overall metrics, an in-depth error analysis was conducted to understand where prediction inaccuracies occur.

Key observations include:

- Predictions remain highly accurate for the majority of standard routes.
- Larger errors are primarily associated with premium-priced flights.
- Seasonal demand spikes introduce greater pricing volatility.
- Extremely expensive tickets are inherently more difficult to predict due to limited representation in historical data.

This analysis highlights potential areas for future model improvements.

---

# 🧠 Model Explainability

Interpretability is essential for business adoption.

Feature importance analysis was performed to understand which variables contribute most to ticket price prediction.

The most influential features include:

- Airline
- Route
- Flight Duration
- Total Stops
- Journey Month
- Departure Time

Understanding feature importance enables stakeholders to trust the model's predictions and derive meaningful business insights.

---

# 🏷️ Model Card

### Model Name

Flight Ticket Price Predictor

### Model Type

Supervised Machine Learning

### Algorithm

XGBoost Regressor

### Problem Category

Regression

### Target Variable

Ticket Price

### Training Objective

Estimate airline ticket prices based on flight characteristics and travel information.

### Intended Use

- Revenue forecasting
- Pricing analytics
- Travel planning
- Airline market analysis
- Decision support systems

### Limitations

- Predictions depend on historical pricing behavior.
- Sudden market events or external disruptions may reduce prediction accuracy.
- The model is not intended for real-time dynamic pricing without continuous retraining.

---

# 📦 Model Packaging

To facilitate deployment, the trained model was packaged using MLflow.

The packaged model includes:

- Serialized XGBoost Model
- Input/Output Signature
- Environment Dependencies
- Metadata
- Version Information

This ensures seamless portability across different environments.

---

# 🔄 MLflow Experiment Tracking

Every experiment performed during development was automatically logged using MLflow.

Tracked artifacts include:

- Parameters
- Evaluation Metrics
- Model Artifacts
- Feature Importance
- Execution Time
- Source Code Version
- Training Environment

Benefits:

- Complete experiment reproducibility
- Easy comparison between model versions
- Centralized experiment management
- Simplified collaboration

---

# 📚 MLflow Model Registry

Only models meeting predefined quality thresholds were promoted to the registry.

Lifecycle Stages

```text
Training
      │
      ▼
Experiment Tracking
      │
      ▼
Model Validation
      │
      ▼
Quality Gate
      │
      ▼
Model Registry
      │
      ▼
Staging
      │
      ▼
Production
```

This workflow guarantees that only validated models are deployed to production environments.

---

# 🚀 FastAPI Deployment

A lightweight RESTful API was developed using **FastAPI** to expose the trained model for real-time predictions.

### Available Endpoints

| Endpoint | Description |
|----------|-------------|
| POST /predict | Predict ticket price |
| GET /health | Health check |
| GET /model-info | Model metadata |

### Example Request

```json
{
  "Airline": "IndiGo",
  "Source": "Delhi",
  "Destination": "Cochin",
  "Duration": 170,
  "Total_Stops": 1
}
```

### Example Response

```json
{
  "predicted_price": 5824.71,
  "model_version": "Production",
  "status": "success"
}
```

---

# 📊 Interactive Streamlit Dashboard

A user-friendly Streamlit application was developed to make the model accessible to both technical and non-technical users.

Dashboard capabilities include:

- Real-time price prediction
- Interactive data exploration
- Business analytics
- Feature visualization
- Model performance overview
- User-friendly prediction interface

The dashboard transforms complex machine learning outputs into actionable business insights.

---

# 📁 Project Structure

```text
Flight-Ticket-Price-Prediction/
│
├── data/
├── notebooks/
├── preprocessing/
├── models/
├── api/
├── dashboard/
├── mlruns/
├── docs/
├── artifacts/
├── requirements.txt
├── README.md
└── app.py
```

---

# ⚡ Installation & Quick Start

Clone the repository

```bash
git clone https://github.com/your-username/Flight-Ticket-Price-Prediction.git
```

Navigate to the project

```bash
cd Flight-Ticket-Price-Prediction
```

Create a virtual environment

```bash
python -m venv .venv
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the FastAPI service

```bash
uvicorn api:app --reload
```

Launch the Streamlit dashboard

```bash
streamlit run app.py
```

---

