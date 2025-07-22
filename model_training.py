# model_training.py
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("garments_worker_productivity.csv")
df.dropna(inplace=True)

# Encode categorical columns
le_dept = LabelEncoder()
le_day = LabelEncoder()
df['department'] = le_dept.fit_transform(df['department'])
df['day'] = le_day.fit_transform(df['day'])

# Features and Target
X = df[['department', 'day', 'team', 'targeted_productivity', 'smv', 'wip',
        'over_time', 'incentive', 'idle_time', 'idle_men', 'no_of_style_change']]
y = df['actual_productivity']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained and saved as model.pkl")
