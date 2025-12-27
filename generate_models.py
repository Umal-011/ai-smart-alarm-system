import pandas as pd
from utils import preprocess_for_training, train_and_save_models, CSV_PATH

df = pd.read_csv(CSV_PATH)
df = preprocess_for_training(df)
train_and_save_models(df)
print("✅ smart_alarm_models.joblib created in /data")
