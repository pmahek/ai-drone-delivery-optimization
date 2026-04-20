import pandas as pd
import numpy as np

# for reproducibility
np.random.seed(42)

# number of rows
rows = 3000

# generate features
distance = np.random.randint(1, 50, rows)       # distance in km
weight = np.random.uniform(0.5, 10, rows)       # weight in kg
priority = np.random.randint(1, 4, rows)        # 1=low,2=medium,3=high
weather = np.random.randint(0, 2, rows)         # 0=clear,1=bad weather

# realistic delivery time formula
delivery_time = (
    distance * 2
    + weight * 1.5
    + priority * 3
    + weather * 8
    + np.random.normal(0, 3, rows)   # randomness
)

# create dataframe
df = pd.DataFrame({
    "distance_km": distance,
    "weight_kg": weight,
    "priority": priority,
    "weather": weather,
    "delivery_time": delivery_time
})

# ensure delivery time positive
df["delivery_time"] = df["delivery_time"].round(2)

# save dataset
df.to_csv("dataset.csv", index=False)

print("Dataset created successfully with", rows, "rows")