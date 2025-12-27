"""
Data cleaning and repair utilities for the Smart Alarm System.
Handles data quality issues and provides automatic data correction.
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

def detect_data_anomalies(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect various data anomalies in the sleep dataset."""
    anomalies = {
        'invalid_durations': [],
        'missing_values': [],
        'duplicate_dates': [],
        'unrealistic_times': [],
        'outliers': []
    }
    
    for idx, row in df.iterrows():
        # Check for missing critical values
        if pd.isna(row['date']) or pd.isna(row['sleep_time']) or pd.isna(row['wake_time']):
            anomalies['missing_values'].append(idx)
            continue
        
        try:
            # Calculate sleep duration
            sleep_h, sleep_m = map(int, str(row['sleep_time']).split(':'))
            wake_h, wake_m = map(int, str(row['wake_time']).split(':'))
            
            sleep_minutes = sleep_h * 60 + sleep_m
            wake_minutes = wake_h * 60 + wake_m
            
            # Handle overnight sleep
            if wake_minutes >= sleep_minutes:
                duration_hours = (wake_minutes - sleep_minutes) / 60
            else:
                duration_hours = ((24 * 60) - sleep_minutes + wake_minutes) / 60
            
            # Check for unrealistic durations
            if duration_hours < 2 or duration_hours > 15:
                anomalies['invalid_durations'].append({
                    'row': idx, 
                    'duration': duration_hours,
                    'sleep_time': row['sleep_time'],
                    'wake_time': row['wake_time']
                })
            
            # Check for unrealistic sleep times (like 12:45 PM)
            if 6 <= sleep_h <= 18:  # Sleep time during day hours
                anomalies['unrealistic_times'].append({
                    'row': idx,
                    'sleep_time': row['sleep_time'],
                    'issue': 'daytime_sleep'
                })
        
        except (ValueError, AttributeError) as e:
            anomalies['invalid_durations'].append({
                'row': idx, 
                'error': str(e),
                'sleep_time': row['sleep_time'],
                'wake_time': row['wake_time']
            })
    
    # Check for duplicate dates
    duplicate_dates = df[df.duplicated(subset=['date'], keep=False)]
    if not duplicate_dates.empty:
        anomalies['duplicate_dates'] = duplicate_dates.index.tolist()
    
    return anomalies

def suggest_corrections(anomalies: Dict[str, Any], df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Suggest automatic corrections for detected anomalies."""
    corrections = []
    
    for anomaly in anomalies['invalid_durations']:
        if isinstance(anomaly, dict) and 'row' in anomaly:
            row_idx = anomaly['row']
            row = df.iloc[row_idx]
            
            # Try to fix obvious issues
            sleep_time = str(row['sleep_time'])
            wake_time = str(row['wake_time'])
            
            # Check if it's a PM/AM confusion (12:45 -> 00:45)
            if sleep_time.startswith('12:'):
                corrected_sleep = sleep_time.replace('12:', '00:', 1)
                corrections.append({
                    'row': row_idx,
                    'type': 'time_correction',
                    'field': 'sleep_time',
                    'original': sleep_time,
                    'corrected': corrected_sleep,
                    'reason': '12:XX converted to 00:XX (midnight)'
                })
            
            # If wake time is before sleep time and seems like PM confusion
            elif wake_time.startswith('05:') and sleep_time.startswith('12:'):
                corrected_sleep = '00:' + sleep_time[3:]
                corrections.append({
                    'row': row_idx,
                    'type': 'time_correction',
                    'field': 'sleep_time',
                    'original': sleep_time,
                    'corrected': corrected_sleep,
                    'reason': 'Assumed 12:XX PM should be 00:XX AM'
                })
    
    return corrections

def apply_corrections(df: pd.DataFrame, corrections: List[Dict[str, Any]], 
                     auto_apply: bool = False) -> pd.DataFrame:
    """Apply suggested corrections to the dataframe."""
    df_corrected = df.copy()
    applied_corrections = []
    
    for correction in corrections:
        if auto_apply or input(f"Apply correction for row {correction['row']} "
                              f"({correction['reason']})? [y/N]: ").lower() == 'y':
            
            row_idx = correction['row']
            field = correction['field']
            new_value = correction['corrected']
            
            df_corrected.iloc[row_idx, df_corrected.columns.get_loc(field)] = new_value
            applied_corrections.append(correction)
            
            logger.info(f"Applied correction to row {row_idx}: "
                       f"{field} {correction['original']} -> {new_value}")
    
    return df_corrected, applied_corrections

def clean_dataset(csv_path: str, backup: bool = True, auto_correct: bool = False) -> Dict[str, Any]:
    """Comprehensive dataset cleaning with automatic corrections."""
    try:
        # Load data
        df = pd.read_csv(csv_path)
        original_rows = len(df)
        
        # Create backup if requested
        if backup:
            backup_path = f"{csv_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            df.to_csv(backup_path, index=False)
            logger.info(f"Backup created: {backup_path}")
        
        # Detect anomalies
        logger.info("Detecting data anomalies...")
        anomalies = detect_data_anomalies(df)
        
        # Generate corrections
        corrections = suggest_corrections(anomalies, df)
        
        # Apply corrections
        df_cleaned = df.copy()
        applied_corrections = []
        
        if corrections:
            logger.info(f"Found {len(corrections)} potential corrections")
            df_cleaned, applied_corrections = apply_corrections(df, corrections, auto_correct)
        
        # Remove rows with critical issues that can't be corrected
        critical_issues = []
        for anomaly in anomalies['invalid_durations']:
            if isinstance(anomaly, dict) and anomaly.get('duration', 0) > 20:  # > 20 hours
                critical_issues.append(anomaly['row'])
        
        if critical_issues:
            logger.warning(f"Removing {len(critical_issues)} rows with critical issues")
            df_cleaned = df_cleaned.drop(critical_issues)
        
        # Fill missing day names
        if 'day' in df_cleaned.columns:
            df_cleaned['day'] = df_cleaned.apply(
                lambda row: pd.to_datetime(row['date']).strftime('%A') 
                if pd.isna(row['day']) else row['day'], axis=1
            )
        
        # Save cleaned data
        df_cleaned.to_csv(csv_path, index=False)
        
        cleaning_report = {
            'original_rows': original_rows,
            'cleaned_rows': len(df_cleaned),
            'rows_removed': original_rows - len(df_cleaned),
            'anomalies_detected': anomalies,
            'corrections_applied': applied_corrections,
            'critical_issues_removed': critical_issues
        }
        
        logger.info(f"Dataset cleaned: {original_rows} -> {len(df_cleaned)} rows")
        return cleaning_report
        
    except Exception as e:
        logger.error(f"Error cleaning dataset: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    from config import CSV_PATH
    
    print("🧹 Starting dataset cleaning...")
    report = clean_dataset(CSV_PATH, backup=True, auto_correct=True)
    
    print("\n📊 Cleaning Report:")
    print(f"Original rows: {report['original_rows']}")
    print(f"Cleaned rows: {report['cleaned_rows']}")
    print(f"Rows removed: {report['rows_removed']}")
    print(f"Corrections applied: {len(report['corrections_applied'])}")
    
    if report['corrections_applied']:
        print("\nApplied corrections:")
        for correction in report['corrections_applied']:
            print(f"  Row {correction['row']}: {correction['reason']}")