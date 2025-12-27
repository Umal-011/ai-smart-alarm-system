"""
Configuration settings for the Smart Alarm System.
Centralized configuration management with environment variable support.
"""

import os
from typing import Dict, Any
from datetime import timedelta

class Config:
    """Configuration class with environment variable support."""
    
    # File paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
    
    # Data files
    CSV_FILENAME = os.getenv("ALARM_CSV_FILE", "Demo_Sleep_Log__first_200_rows_.csv")
    CSV_PATH = os.path.join(DATA_DIR, CSV_FILENAME)
    MODEL_FILENAME = os.getenv("ALARM_MODEL_FILE", "smart_alarm_models.joblib")
    MODEL_PATH = os.path.join(DATA_DIR, MODEL_FILENAME)
    
    # Machine Learning settings
    ML_CONFIG = {
        'n_estimators': int(os.getenv("ML_N_ESTIMATORS", "100")),
        'random_state': int(os.getenv("ML_RANDOM_STATE", "42")),
        'max_depth': int(os.getenv("ML_MAX_DEPTH", "10")),
        'min_samples_split': int(os.getenv("ML_MIN_SAMPLES_SPLIT", "5")),
        'min_samples_leaf': int(os.getenv("ML_MIN_SAMPLES_LEAF", "2"))
    }
    
    # Validation settings
    VALIDATION_CONFIG = {
        'min_sleep_duration_hours': float(os.getenv("MIN_SLEEP_HOURS", "3.0")),
        'max_sleep_duration_hours': float(os.getenv("MAX_SLEEP_HOURS", "12.0")),
        'min_dataset_size': int(os.getenv("MIN_DATASET_SIZE", "10")),
        'max_buffer_minutes': int(os.getenv("MAX_BUFFER_MINUTES", "120")),
        'unusual_bedtime_start': int(os.getenv("UNUSUAL_BEDTIME_START", "6")),
        'unusual_bedtime_end': int(os.getenv("UNUSUAL_BEDTIME_END", "18"))
    }
    
    # Security settings
    SECURITY_CONFIG = {
        'max_string_length': int(os.getenv("MAX_STRING_LENGTH", "100")),
        'allowed_file_extensions': {'.csv', '.joblib', '.backup'},
        'enable_backups': os.getenv("ENABLE_BACKUPS", "true").lower() == "true",
        'max_backup_files': int(os.getenv("MAX_BACKUP_FILES", "5"))
    }
    
    # Logging settings
    LOG_CONFIG = {
        'level': os.getenv("LOG_LEVEL", "INFO"),
        'format': os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        'enable_file_logging': os.getenv("ENABLE_FILE_LOGGING", "false").lower() == "true",
        'log_file': os.path.join(DATA_DIR, "smart_alarm.log")
    }
    
    # Streamlit settings
    UI_CONFIG = {
        'page_title': os.getenv("UI_PAGE_TITLE", "Smart Alarm System"),
        'page_icon': os.getenv("UI_PAGE_ICON", "⏰"),
        'layout': os.getenv("UI_LAYOUT", "wide"),
        'theme_primary_color': os.getenv("UI_PRIMARY_COLOR", "#1f77b4"),
        'max_recent_entries': int(os.getenv("UI_MAX_RECENT_ENTRIES", "10")),
        'chart_height': int(os.getenv("UI_CHART_HEIGHT", "300"))
    }
    
    # Performance settings
    PERFORMANCE_CONFIG = {
        'cache_predictions': os.getenv("CACHE_PREDICTIONS", "true").lower() == "true",
        'cache_timeout_minutes': int(os.getenv("CACHE_TIMEOUT_MINUTES", "30")),
        'batch_size': int(os.getenv("BATCH_SIZE", "1000")),
        'enable_parallel_processing': os.getenv("ENABLE_PARALLEL", "false").lower() == "true"
    }
    
    @classmethod
    def get_config_summary(cls) -> Dict[str, Any]:
        """Get a summary of current configuration."""
        return {
            'csv_path': cls.CSV_PATH,
            'model_path': cls.MODEL_PATH,
            'ml_config': cls.ML_CONFIG,
            'validation_config': cls.VALIDATION_CONFIG,
            'security_config': cls.SECURITY_CONFIG,
            'log_level': cls.LOG_CONFIG['level'],
            'ui_config': cls.UI_CONFIG,
            'performance_config': cls.PERFORMANCE_CONFIG
        }
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration settings."""
        errors = []
        warnings = []
        
        # Check file paths
        if not os.path.exists(cls.DATA_DIR):
            try:
                os.makedirs(cls.DATA_DIR, exist_ok=True)
                warnings.append(f"Created data directory: {cls.DATA_DIR}")
            except Exception as e:
                errors.append(f"Cannot create data directory: {e}")
        
        # Check CSV file
        if not os.path.exists(cls.CSV_PATH):
            warnings.append(f"CSV file not found: {cls.CSV_PATH}")
        
        # Validate ML config
        if cls.ML_CONFIG['n_estimators'] < 10:
            warnings.append("Low n_estimators may reduce prediction accuracy")
        
        # Validate validation config
        if cls.VALIDATION_CONFIG['min_sleep_duration_hours'] >= cls.VALIDATION_CONFIG['max_sleep_duration_hours']:
            errors.append("min_sleep_duration_hours must be less than max_sleep_duration_hours")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

# Global configuration instance
config = Config()

# Export commonly used paths and settings
CSV_PATH = config.CSV_PATH
MODEL_PATH = config.MODEL_PATH
ML_CONFIG = config.ML_CONFIG
VALIDATION_CONFIG = config.VALIDATION_CONFIG
SECURITY_CONFIG = config.SECURITY_CONFIG
UI_CONFIG = config.UI_CONFIG