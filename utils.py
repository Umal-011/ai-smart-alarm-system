import pandas as pd
import joblib
import os
import logging
import warnings
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Tuple, Any
import shutil

# Import validators
from validators import (
    ValidationError, SecurityError, validate_time_string, validate_dataframe,
    validate_prediction_input, sanitize_user_input, check_file_security
)


# Suppress sklearn warnings about feature names
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import configuration
from config import CSV_PATH, MODEL_PATH, ML_CONFIG, VALIDATION_CONFIG

# ----------------------------
# Utility
# ----------------------------
def to_minutes(tstr):
    h, m = map(int, tstr.split(":"))
    return h * 60 + m

def preprocess_for_training(df):
    df["sleep_time_minutes"] = df["sleep_time"].apply(to_minutes)
    df["wake_time_minutes"] = df["wake_time"].apply(to_minutes)
    return df

# ----------------------------
# ML Training & Prediction
# ----------------------------
def train_day_model(df: pd.DataFrame) -> RandomForestRegressor:
    """Train a model based on day of week patterns."""
    try:
        df_copy = df.copy()
        df_copy["day_of_week"] = pd.to_datetime(df_copy["date"]).dt.dayofweek
        
        # Use proper column names to avoid sklearn warnings
        feature_columns = ["day_of_week", "sleep_time_minutes"]
        X = df_copy[feature_columns]
        y = df_copy["wake_time_minutes"]
        
        model = RandomForestRegressor(
            n_estimators=ML_CONFIG['n_estimators'],
            random_state=ML_CONFIG['random_state'],
            max_depth=ML_CONFIG['max_depth'],
            min_samples_split=ML_CONFIG['min_samples_split'],
            min_samples_leaf=ML_CONFIG['min_samples_leaf']
        )
        model.fit(X, y)
        
        logger.info(f"Day model trained with {len(df_copy)} samples")
        return model
    except Exception as e:
        logger.error(f"Error training day model: {e}")
        raise

def train_week_model(df: pd.DataFrame) -> RandomForestRegressor:
    """Train a model based on weekly patterns."""
    try:
        df_copy = df.copy()
        df_copy["week_number"] = pd.to_datetime(df_copy["date"]).dt.isocalendar().week
        
        # Use proper column names to avoid sklearn warnings
        feature_columns = ["week_number", "sleep_time_minutes"]
        X = df_copy[feature_columns]
        y = df_copy["wake_time_minutes"]
        
        model = RandomForestRegressor(
            n_estimators=ML_CONFIG['n_estimators'],
            random_state=ML_CONFIG['random_state'],
            max_depth=ML_CONFIG['max_depth'],
            min_samples_split=ML_CONFIG['min_samples_split'],
            min_samples_leaf=ML_CONFIG['min_samples_leaf']
        )
        model.fit(X, y)
        
        logger.info(f"Week model trained with {len(df_copy)} samples")
        return model
    except Exception as e:
        logger.error(f"Error training week model: {e}")
        raise

def train_and_save_models(df: pd.DataFrame, save_path: str = MODEL_PATH) -> Dict[str, RandomForestRegressor]:
    """Train and save both day and week models."""
    try:
        logger.info("Training day and week models...")
        day_model = train_day_model(df)
        week_model = train_week_model(df)
        
        models = {"day": day_model, "week": week_model}
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save models
        joblib.dump(models, save_path)
        logger.info(f"Models saved to {save_path}")
        
        return models
        
    except Exception as e:
        logger.error(f"Error training/saving models: {e}")
        raise

def load_models(path: str = MODEL_PATH) -> Dict[str, RandomForestRegressor]:
    """Load pre-trained models."""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        models = joblib.load(path)
        
        # Validate loaded models
        if not isinstance(models, dict) or "day" not in models or "week" not in models:
            raise ValueError("Invalid model file structure")
        
        logger.info(f"Models loaded from {path}")
        return models
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

# Use the comprehensive validator from validators module
validate_time_format = validate_time_string

def predict_wakeup(models: Dict[str, RandomForestRegressor], sleep_time_str: str) -> Dict[str, Any]:
    """Predict wake up time using ensemble of day and week models."""
    try:
        # Comprehensive input validation
        validation_result = validate_prediction_input(sleep_time_str)
        if not validation_result['is_valid']:
            raise ValidationError(f"Input validation failed: {validation_result['errors']}")
        
        # Log warnings if any
        for warning in validation_result['warnings']:
            logger.warning(warning)
        
        sleep_h, sleep_m = map(int, sleep_time_str.split(":"))
        sleep_minutes = sleep_h * 60 + sleep_m

        today = pd.Timestamp.now()
        day_of_week = today.dayofweek
        week_number = today.isocalendar().week

        # Create proper feature arrays with column names
        day_features = pd.DataFrame([[day_of_week, sleep_minutes]], 
                                  columns=["day_of_week", "sleep_time_minutes"])
        week_features = pd.DataFrame([[week_number, sleep_minutes]], 
                                   columns=["week_number", "sleep_time_minutes"])

        pred_day = models["day"].predict(day_features)[0]
        pred_week = models["week"].predict(week_features)[0]
        pred_final = (pred_day + pred_week) / 2

        # Ensure wake time is within 24-hour format
        hh = int(pred_final // 60) % 24
        mm = int(pred_final % 60)
        
        # Calculate sleep duration for validation
        sleep_duration = (pred_final - sleep_minutes) % 1440
        
        # Quality checks
        quality_warnings = []
        if sleep_duration < 240:  # Less than 4 hours
            quality_warnings.append("Very short sleep duration predicted")
        elif sleep_duration > 720:  # More than 12 hours
            quality_warnings.append("Very long sleep duration predicted")
        
        # Confidence calculation based on model agreement
        model_diff = abs(pred_day - pred_week)
        confidence = "high" if model_diff < 30 else "medium" if model_diff < 60 else "low"
        
        result = {
            "pred_minutes": pred_final,
            "pred_meta_hhmm": f"{hh:02d}:{mm:02d}",
            "sleep_duration_hours": sleep_duration / 60,
            "day_prediction": pred_day,
            "week_prediction": pred_week,
            "confidence": confidence,
            "model_agreement": model_diff,
            "quality_warnings": quality_warnings
        }
        
        logger.info(f"Prediction for {sleep_time_str}: {result['pred_meta_hhmm']} "
                   f"({result['sleep_duration_hours']:.1f}h, confidence: {confidence})")
        return result
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Error predicting wake time for {sleep_time_str}: {e}")
        raise

# ----------------------------
# Sleep Log Helpers
# ----------------------------
def create_backup(csv_path: str) -> str:
    """Create a backup of the CSV file."""
    try:
        backup_path = f"{csv_path}.backup"
        shutil.copy2(csv_path, backup_path)
        return backup_path
    except Exception as e:
        logger.warning(f"Could not create backup: {e}")
        return ""

def log_new_sleep_entry(csv_path: str, sleep_time_str: str, models: Dict[str, RandomForestRegressor], buffer_minutes: int = 15) -> Tuple[Dict, Dict]:
    """Log a new sleep entry with prediction."""
    try:
        # Security check for file path
        check_file_security(csv_path)
        
        # Validate inputs using comprehensive validation
        validation_result = validate_prediction_input(sleep_time_str, buffer_minutes)
        if not validation_result['is_valid']:
            raise ValidationError(f"Input validation failed: {validation_result['errors']}")
        
        # Log warnings
        for warning in validation_result['warnings']:
            logger.warning(warning)
        
        # Create backup before modifying
        create_backup(csv_path)
        
        fmt = "%H:%M"
        sleep_dt = datetime.strptime(sleep_time_str, fmt)
        adjusted_sleep = (sleep_dt + timedelta(minutes=buffer_minutes)).strftime(fmt)

        result = predict_wakeup(models, adjusted_sleep)
        wake_time = result["pred_meta_hhmm"]

        today_str = datetime.now().date().isoformat()
        
        # Check if entry for today already exists
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            existing_today = df[df["date"] == today_str]
            if not existing_today.empty:
                logger.warning(f"Entry for {today_str} already exists. Updating instead of creating new.")
                return update_sleep_entry(csv_path, today_str, adjusted_sleep, wake_time), result
        else:
            # Create new file with header
            df = pd.DataFrame(columns=["date", "sleep_time", "wake_time", "mood", "day"])
        
        new_row = {
            "date": today_str,
            "sleep_time": adjusted_sleep,
            "wake_time": wake_time,
            "mood": "",
            "day": datetime.now().strftime("%A")
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_path, index=False)
        
        logger.info(f"New sleep entry logged: {today_str} - {adjusted_sleep} -> {wake_time}")
        return new_row, result
        
    except Exception as e:
        logger.error(f"Error logging sleep entry: {e}")
        raise

def update_sleep_entry(csv_path: str, date_str: str, sleep_time: str, wake_time: str) -> Dict:
    """Update existing sleep entry."""
    try:
        df = pd.read_csv(csv_path)
        mask = df["date"] == date_str
        if not mask.any():
            raise ValueError(f"No entry found for {date_str}")
        
        df.loc[mask, "sleep_time"] = sleep_time
        df.loc[mask, "wake_time"] = wake_time
        df.to_csv(csv_path, index=False)
        
        return df.loc[mask].to_dict(orient="records")[0]
    except Exception as e:
        logger.error(f"Error updating sleep entry: {e}")
        raise

def update_mood(csv_path: str, date_str: str, mood_value: str) -> Dict:
    """Update mood for a specific date."""
    try:
        # Validate mood value
        valid_moods = ["fresh", "sleepy", ""]
        if mood_value not in valid_moods:
            raise ValueError(f"Invalid mood value: {mood_value}. Must be one of {valid_moods}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Create backup
        create_backup(csv_path)
        
        df = pd.read_csv(csv_path)
        mask = df["date"] == date_str
        if not mask.any():
            raise ValueError(f"No entry found for {date_str}")
        
        df.loc[mask, "mood"] = mood_value
        df.to_csv(csv_path, index=False)
        
        updated_entry = df.loc[mask].to_dict(orient="records")[0]
        logger.info(f"Mood updated for {date_str}: {mood_value}")
        return updated_entry
        
    except Exception as e:
        logger.error(f"Error updating mood: {e}")
        raise

# ----------------------------
# Setup
# ----------------------------
def validate_csv_data(df: pd.DataFrame) -> bool:
    """Validate the CSV data structure and content using comprehensive validators."""
    try:
        # Use the comprehensive dataframe validator
        validation_result = validate_dataframe(df)
        
        # Log all warnings
        for warning in validation_result['warnings']:
            logger.warning(warning)
        
        # Handle errors
        if not validation_result['is_valid']:
            error_msg = f"CSV validation failed with {len(validation_result['errors'])} errors:\n"
            for error in validation_result['errors'][:10]:  # Show first 10 errors
                error_msg += f"  - {error}\n"
            if len(validation_result['errors']) > 10:
                error_msg += f"  ... and {len(validation_result['errors']) - 10} more errors"
            
            raise ValueError(error_msg)
        
        logger.info(f"CSV validation passed: {validation_result['valid_rows']}/{validation_result['total_rows']} valid rows")
        return True
        
    except Exception as e:
        logger.error(f"CSV validation error: {e}")
        raise

def ensure_setup() -> Tuple[pd.DataFrame, Dict[str, RandomForestRegressor]]:
    """Setup data and models with comprehensive validation."""
    try:
        # Check if CSV exists
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"CSV not found at {CSV_PATH}")
        
        # Load and validate data
        logger.info(f"Loading data from {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
        logger.info(f"Loaded {len(df)} sleep records")
        
        # Validate data
        validate_csv_data(df)
        
        # Preprocess data
        df = preprocess_for_training(df)
        
        # Handle models
        if not os.path.exists(MODEL_PATH):
            logger.info("Training new models...")
            models = train_and_save_models(df, MODEL_PATH)
            logger.info("✅ Models trained & saved")
        else:
            logger.info("Loading existing models...")
            models = load_models(MODEL_PATH)
            logger.info("✅ Models loaded")
        
        # Validate models
        if not models or "day" not in models or "week" not in models:
            raise ValueError("Invalid models loaded. Missing 'day' or 'week' model.")
        
        logger.info("Setup completed successfully")
        return df, models
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        raise
