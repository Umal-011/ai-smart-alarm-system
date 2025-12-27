"""
Data validation and security utilities for the Smart Alarm System.
Provides comprehensive input validation, data sanitization, and security checks.
"""

import pandas as pd
import re
from datetime import datetime, time, date
from typing import Union, Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

class SecurityError(Exception):
    """Custom exception for security-related issues."""
    pass

def sanitize_string(input_str: str, max_length: int = 100) -> str:
    """Sanitize string input to prevent injection attacks."""
    if not isinstance(input_str, str):
        raise ValidationError(f"Expected string, got {type(input_str)}")
    
    # Remove potential dangerous characters
    sanitized = re.sub(r'[<>"\';\\&]', '', input_str.strip())
    
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
        logger.warning(f"String truncated to {max_length} characters")
    
    return sanitized

def validate_time_string(time_str: str) -> bool:
    """Validate time string format (HH:MM)."""
    if not isinstance(time_str, str):
        return False
    
    # Check format
    pattern = re.compile(r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$')
    if not pattern.match(time_str):
        return False
    
    try:
        hours, minutes = map(int, time_str.split(':'))
        return 0 <= hours <= 23 and 0 <= minutes <= 59
    except ValueError:
        return False

def validate_date_string(date_str: str) -> bool:
    """Validate date string format (YYYY-MM-DD)."""
    if not isinstance(date_str, str):
        return False
    
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def validate_mood_value(mood: str) -> bool:
    """Validate mood input."""
    if not isinstance(mood, str):
        return False
    
    valid_moods = {'fresh', 'sleepy', ''}
    return mood.lower().strip() in valid_moods

def validate_sleep_duration(sleep_time: str, wake_time: str) -> Dict[str, Any]:
    """Validate sleep duration and detect anomalies."""
    if not validate_time_string(sleep_time) or not validate_time_string(wake_time):
        raise ValidationError("Invalid time format")
    
    # Convert to minutes
    sleep_h, sleep_m = map(int, sleep_time.split(':'))
    wake_h, wake_m = map(int, wake_time.split(':'))
    
    sleep_minutes = sleep_h * 60 + sleep_m
    wake_minutes = wake_h * 60 + wake_m
    
    # Calculate duration (handle overnight sleep)
    if wake_minutes >= sleep_minutes:
        duration = wake_minutes - sleep_minutes
    else:
        duration = (24 * 60) - sleep_minutes + wake_minutes
    
    # Validate duration
    warnings = []
    errors = []
    
    if duration < 180:  # Less than 3 hours
        errors.append(f"Sleep duration too short: {duration/60:.1f} hours")
    elif duration < 300:  # Less than 5 hours
        warnings.append(f"Short sleep duration: {duration/60:.1f} hours")
    
    if duration > 720:  # More than 12 hours
        errors.append(f"Sleep duration too long: {duration/60:.1f} hours")
    elif duration > 600:  # More than 10 hours
        warnings.append(f"Long sleep duration: {duration/60:.1f} hours")
    
    return {
        'duration_minutes': duration,
        'duration_hours': duration / 60,
        'warnings': warnings,
        'errors': errors,
        'is_valid': len(errors) == 0
    }

def validate_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a single CSV row."""
    errors = []
    warnings = []
    
    # Required fields
    required_fields = ['date', 'sleep_time', 'wake_time']
    for field in required_fields:
        if field not in row or not row[field]:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return {'is_valid': False, 'errors': errors, 'warnings': warnings}
    
    # Validate date
    if not validate_date_string(str(row['date'])):
        errors.append(f"Invalid date format: {row['date']}")
    
    # Validate times
    if not validate_time_string(str(row['sleep_time'])):
        errors.append(f"Invalid sleep time format: {row['sleep_time']}")
    
    if not validate_time_string(str(row['wake_time'])):
        errors.append(f"Invalid wake time format: {row['wake_time']}")
    
    # Validate mood if present
    if 'mood' in row and row['mood'] and not validate_mood_value(str(row['mood'])):
        errors.append(f"Invalid mood value: {row['mood']}")
    
    # If basic validation passes, check sleep duration
    if not errors:
        try:
            duration_check = validate_sleep_duration(str(row['sleep_time']), str(row['wake_time']))
            warnings.extend(duration_check['warnings'])
            if not duration_check['is_valid']:
                errors.extend(duration_check['errors'])
        except Exception as e:
            errors.append(f"Duration validation failed: {e}")
    
    return {
        'is_valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate entire dataframe."""
    if df.empty:
        return {'is_valid': False, 'errors': ['Dataframe is empty'], 'warnings': []}
    
    total_errors = []
    total_warnings = []
    invalid_rows = []
    
    for idx, row in df.iterrows():
        validation = validate_csv_row(row.to_dict())
        
        if not validation['is_valid']:
            invalid_rows.append(idx)
            for error in validation['errors']:
                total_errors.append(f"Row {idx}: {error}")
        
        for warning in validation['warnings']:
            total_warnings.append(f"Row {idx}: {warning}")
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['date']).sum()
    if duplicates > 0:
        total_warnings.append(f"Found {duplicates} duplicate date entries")
    
    # Check data consistency
    if len(df) < 5:
        total_warnings.append("Very small dataset - predictions may be unreliable")
    
    return {
        'is_valid': len(total_errors) == 0,
        'errors': total_errors,
        'warnings': total_warnings,
        'invalid_rows': invalid_rows,
        'total_rows': len(df),
        'valid_rows': len(df) - len(invalid_rows)
    }

def sanitize_user_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize user input data."""
    sanitized = {}
    
    for key, value in data.items():
        if isinstance(value, str):
            sanitized[key] = sanitize_string(value)
        elif isinstance(value, (int, float)) and -10000 < value < 10000:  # Reasonable bounds
            sanitized[key] = value
        elif value is None:
            sanitized[key] = ""
        else:
            logger.warning(f"Skipping potentially unsafe input: {key}={value}")
            sanitized[key] = ""
    
    return sanitized

def check_file_security(file_path: str) -> bool:
    """Basic security check for file paths."""
    import os
    
    # Normalize path
    normalized_path = os.path.normpath(file_path)
    
    # Check for path traversal attempts
    if '..' in normalized_path or normalized_path.startswith('/'):
        raise SecurityError(f"Potentially unsafe file path: {file_path}")
    
    # Check file extension
    allowed_extensions = {'.csv', '.joblib', '.backup'}
    _, ext = os.path.splitext(normalized_path)
    if ext not in allowed_extensions:
        logger.warning(f"Unusual file extension: {ext}")
    
    return True

def validate_prediction_input(sleep_time: str, buffer_minutes: int = 15) -> Dict[str, Any]:
    """Validate input for prediction functions."""
    errors = []
    warnings = []
    
    # Validate sleep time
    if not validate_time_string(sleep_time):
        errors.append(f"Invalid sleep time format: {sleep_time}")
    
    # Validate buffer
    if not isinstance(buffer_minutes, int) or not (0 <= buffer_minutes <= 120):
        errors.append(f"Buffer minutes must be integer between 0-120: {buffer_minutes}")
    
    # Check if it's a reasonable bedtime
    if not errors:
        try:
            hour = int(sleep_time.split(':')[0])
            if 6 <= hour <= 18:  # 6 AM to 6 PM
                warnings.append(f"Unusual bedtime: {sleep_time} (daytime)")
        except:
            pass
    
    return {
        'is_valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

# Export main validation functions
__all__ = [
    'ValidationError',
    'SecurityError',
    'validate_time_string',
    'validate_date_string',
    'validate_mood_value',
    'validate_sleep_duration',
    'validate_csv_row',
    'validate_dataframe',
    'sanitize_user_input',
    'check_file_security',
    'validate_prediction_input'
]