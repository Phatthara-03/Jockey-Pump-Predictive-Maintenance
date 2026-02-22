import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ==============================
# STEP 1: Load Data
# ==============================

df = pd.read_csv("Jockey_cleaned.csv")
df.columns = df.columns.str.strip()

print("Data Loaded")
print(df.head())

# ==============================
# STEP 2: Create RMS
# ==============================

window_size = 200

df["rms"] = (
    df["Amplitude"]
    .rolling(window=window_size)
    .apply(lambda x: np.sqrt(np.mean(x**2)))
)

df = df.dropna().reset_index(drop=True)

# ==============================
# STEP 3: Simulate Degradation
# ==============================

# gradual linear degradation
degradation_trend = np.linspace(0, 4, len(df))
df["rms_degraded"] = df["rms"] + degradation_trend

# acceleration in last 20%
start_accel = int(len(df) * 0.8)
df.loc[start_accel:, "rms_degraded"] += np.linspace(
    0, 2, len(df) - start_accel
)

# ==============================
# STEP 4: ISO 10816-3 Zone Classification
# ==============================

def iso_zone(value):
    if value < 2.8:
        return "Green"
    elif value < 4.5:
        return "Yellow"
    elif value < 7.1:
        return "Orange"
    else:
        return "Red"

df["Zone"] = df["rms_degraded"].apply(iso_zone)

print("\nCurrent Machine Zone:", df["Zone"].iloc[-1])

# ==============================
# STEP 5: Train Polynomial Model
# ==============================

X = np.arange(len(df)).reshape(-1, 1)
y = df["rms_degraded"].values

model = make_pipeline(
    PolynomialFeatures(2),
    LinearRegression()
)

model.fit(X, y)

r2 = model.score(X, y)

print("\nModel Accuracy (Full Fit):")
print("R2:", round(r2, 4))

# ==============================
# STEP 6: Forecast Future
# ==============================

future_steps = 300
X_future = np.arange(len(df), len(df) + future_steps).reshape(-1, 1)
future_forecast = model.predict(X_future)

# ==============================
# STEP 7: Failure Prediction (Red Zone)
# ==============================

threshold = 7.1
failure_index = None

for i, value in enumerate(future_forecast):
    if value > threshold:
        failure_index = i
        break

if failure_index is not None:
    print("\n⚠ Predicted FAILURE at future step:", failure_index)
    print("Estimated Remaining Samples (RUL):", failure_index)
else:
    print("\nNo failure predicted in forecast window")

# ==============================
# STEP 8: Plot Results
# ==============================

plt.figure(figsize=(12,6))

# plot degraded RMS
plt.plot(df["rms_degraded"], label="Simulated RMS")

# plot forecast
plt.plot(
    range(len(df), len(df) + future_steps),
    future_forecast,
    label="Forecast"
)

# ISO zone lines
plt.axhline(2.8, linestyle='--', label="Green/Yellow", color='Green')
plt.axhline(4.5, linestyle='--', label="Yellow/Orange", color='Orange')
plt.axhline(7.1, linestyle='--', label="Critical (Red)", color='red')

plt.legend()
plt.title("Predictive Maintenance - Polynomial Regression (ISO 10816-3)")
plt.xlabel("Samples")
plt.ylabel("RMS (mm/s)")
plt.grid(True)

plt.show()

print("\nSystem Completed")