# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib # Adding this library for streamlit assignment

# Loading the data into a dataframe
file_path = "hotel_bookings.csv"
df = pd.read_csv(file_path)

# Adding a feature for total nights which was not in the dataset yet
df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]

# Selecting the features and the target array
features = ["lead_time", "previous_cancellations", "total_nights", "total_of_special_requests"]
target = "is_canceled"

scaling_info = {}

# New scaling function for streamlit assignment
def scale_features(data):
    scaled_data = {}
    for feature in features:
        mean = scaling_info[feature]["mean"]
        std = scaling_info[feature]["std"]
        scaled_data[feature] = (data[feature] - mean) / std
    return scaled_data

# Scaling with plain Python, not using StandardScaler
for feature in features:
    mean = df[feature].mean()
    std = df[feature].std()
    scaling_info[feature] = {"mean": mean, "std": std}
    df[feature] = [(x - mean) / std for x in df[feature]]

# Splitting df into features (X) and target (y)
X = df[features]
y = df[target]

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Training logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluating the models' performance based on accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model and scaling info for sreamlit assignment
joblib.dump(model, "hotel_cancelation_model.joblib")
joblib.dump(scaling_info, "scaling_info.joblib")
print("Model and scaling info saved")
