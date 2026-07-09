import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger("Model")

def train_model(df, n_estimators, max_depth, learning_rate):
    logger.info('=========== Build ML Model ==========')

    # ====================================================
    # STEP 1: The Firewall (Anti-Leakage Data Split)
    # ====================================================
    logger.info("STEP 1: Applying Firewall - Splitting data into Train and Test sets...")
    # Separate independent variables and target variable first
    X_raw = df.drop(columns=['Price']).copy()
    y_raw = df['Price'].copy()
    
    # Strictly split first to ensure test set remains untouched
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

    # Handling the Total_Stops missing values safely using Train Mode to avoid data leakage
    train_stops_mode = X_train_raw["Total_Stops"].mode()[0]
    X_train_raw["Total_Stops"] = X_train_raw["Total_Stops"].fillna(train_stops_mode)
    X_test_raw["Total_Stops"] = X_test_raw["Total_Stops"].fillna(train_stops_mode)


    # ====================================================
    # STEP 2: Feature Selection
    # ====================================================
    logger.info("STEP 2: Executing Feature Selection...")
    categorical_features = ['Airline', 'Source', 'Destination', 'Dep_Session']
    numeric_features = ['Total_Stops', 'Dep_hour', 'Arrival_hour', 'Duration_mins',
                        'Month_of_Journey', 'Days_of_Journey', 'Day_of_Week',
                        'is_weekend', 'Is_Long_Flight', 'is_peak_season']

    # Filter columns based on selected features
    selected_columns = categorical_features + numeric_features
    X_train = X_train_raw[selected_columns].copy()
    X_test = X_test_raw[selected_columns].copy()


    # ====================================================
    # STEP 3: Handling Outliers (On Train Set Target)
    # ====================================================
    logger.info("STEP 3: Detecting and removing target outliers from Train set...")
    Q1 = y_train.quantile(0.25)
    Q3 = y_train.quantile(0.75)
    IQR = Q3 - Q1
    Lower = Q1 - 1.5 * IQR
    Upper = Q3 + 1.5 * IQR

    # Create mask based on safe target boundaries
    train_keep_mask = (y_train >= Lower) & (y_train <= Upper)
    
    # Filter out rows from both X_train and y_train simultaneously
    X_train = X_train[train_keep_mask]
    y_train = y_train[train_keep_mask]
    logger.info(f"Train dataset shape after outlier removal: {X_train.shape}")


    # ====================================================
    # STEP 4: Handling Skew (Log Transformation)
    # ====================================================
    logger.info("STEP 4: Applying Log Transformation to target variable (y_train & y_test)...")
    y_train = np.log1p(y_train)
    y_test = np.log1p(y_test)


    # ====================================================
    # STEP 5: Encoding & Pipeline Configuration
    # ====================================================
    logger.info("STEP 5: Configuring Pipeline with ColumnTransformer Encoding...")
    
    # Setup OneHotEncoder to automatically process categorical features inside the pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ], remainder='passthrough')

    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', xgb)
    ])


    # ====================================================
    # STEP 6: Model Training & Tuning (The Continuation)
    # ====================================================
    logger.info("STEP 6: Running Cross-Validation & Hyperparameter Tuning...")
    
    # Baseline Cross Validation Check
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')
    logger.info(f"CV MAE Mean (Log Scale): {-cv_results.mean():.4f}")
    logger.info(f"CV MAE Std (Stability): {cv_results.std():.4f}")

    # Calculate weights based on target scale to balance training sample density
    weights = (y_train - y_train.min()) / (y_train.max() - y_train.min()) + 1

    # Hyperparameter search setup: Random Search
    param_dist = {
        'model__n_estimators': [500, 1000, n_estimators],
        'model__learning_rate': [0.01, learning_rate, 0.1],
        'model__max_depth': [max_depth, 6, 9],
        'model__subsample': [0.8, 0.9],
        'model__colsample_bytree': [0.8, 0.9],
        'model__gamma': [0, 1, 5]
    }
    
    random_search = RandomizedSearchCV(pipeline, param_dist, n_iter=10, cv=kf, scoring='r2', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train, model__sample_weight=weights)
    logger.info(f"Best Params from RandomSearch: {random_search.best_params_}")

    # Fine tuning setup: Grid Search
    param_grid = {
        'model__n_estimators': [n_estimators],
        'model__max_depth': [max_depth],
        'model__learning_rate': [learning_rate],
        'model__gamma': [0.5, 1, 1.5],
        'model__subsample': [0.8],
        'model__colsample_bytree': [0.8]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train, model__sample_weight=weights)
    logger.info(f"Best Params from GridSearch: {grid_search.best_params_}")

    # Fit final estimator using the absolute best configuration
    final_model = grid_search.best_estimator_
    final_model.fit(X_train, y_train, model__sample_weight=weights)


    # ====================================================
    # STEP 7: Performance Evaluation & Inference
    # ====================================================
    logger.info("STEP 7: Performing Final Evaluation on Test Set...")
    
    # Predict on unseen test features
    y_pred_log = final_model.predict(X_test)

    # Invert the Log Transformation via exponential mapping for real-world price evaluation
    y_test_original = np.expm1(y_test)
    y_pred_original = np.expm1(y_pred_log)
    y_pred_original = np.maximum(y_pred_original, 0) # Clip any negative values safely

    # Calculate continuous regression metrics
    rmsle = np.sqrt(mean_squared_log_error(y_test_original, y_pred_original))
    mae = mean_absolute_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test, y_pred_log)

    logger.info("-" * 40)
    logger.info(" FINAL EVALUATION RESULTS")
    logger.info("-" * 40)
    logger.info(f"RMSLE: {rmsle:.4f}")
    logger.info(f"MAE (Actual Price): {mae:.2f} Units")
    logger.info(f"R-Squared (Log Domain): {r2:.4%}")
    logger.info("-" * 40)


    # ====================================================
    # STEP 8: Feature Importance Visual Tracking
    # ====================================================
    logger.info("STEP 8: Generating Feature Importance Report...")
    ohe_columns = list(final_model.named_steps['preprocessor']
                       .named_transformers_['cat']
                       .get_feature_names_out(categorical_features))
    all_features = ohe_columns + numeric_features
    importances = final_model.named_steps['model'].feature_importances_

    feature_importance_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Top 10 Features Driving Ticket Prices')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.show()
    plt.close()

    return {
        "model": final_model,
        "rmsle": rmsle,
        "mae": mae,
        "r2": r2,
        "cv_mae_mean": -cv_results.mean(),
        "cv_mae_std": cv_results.std(),
        "params": grid_search.best_params_,
        "search_space_random": param_dist,
        "search_space_grid": param_grid,
        "feature_importance_df": feature_importance_df,
        "X_train": X_train,
        "X": X_raw[selected_columns]
    }
