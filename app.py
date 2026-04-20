import streamlit as st
import pickle
import numpy as np
import os

# ================================
# LOAD MODELS (SAFE CHECK)
# ================================
if not os.path.exists("random_forest.pkl"):
    st.error("⚠️ Random Forest model not found. Please run: python src/model_training.py")
    st.stop()

kmeans = pickle.load(open("kmeans.pkl", "rb"))
rf = pickle.load(open("random_forest.pkl", "rb"))
# ================================
# LOAD MODELS
# ================================
kmeans = pickle.load(open("kmeans.pkl", "rb"))
rf = pickle.load(open("random_forest.pkl", "rb"))

st.set_page_config(layout="wide")
st.title("🚁 AI Drone Delivery Optimization System")

# ================================
# CHARGING STATIONS (SIMULATION)
# ================================
charging_stations = [
    {"name": "Station Alpha", "distance": 5},
    {"name": "Station Beta", "distance": 10},
    {"name": "Station Gamma", "distance": 15}
]

def nearest_station(distance):
    return min(charging_stations, key=lambda x: abs(x["distance"] - distance))

# ================================
# INPUT: DELIVERIES
# ================================
st.header("📦 Enter Deliveries")

num_deliveries = st.slider("Number of Deliveries", 1, 6, 3)

deliveries = []

for i in range(num_deliveries):
    with st.expander(f"📦 Delivery {i+1}", expanded=True):

        col1, col2 = st.columns(2)

        with col1:
            d = st.number_input("Distance (km)", 1, 50, key=f"d{i}")
            w = st.number_input("Weight (kg)", 0.5, 10.0, key=f"w{i}")

        with col2:
            p = st.selectbox("Priority", ["Low","Medium","High"], key=f"p{i}")
            we = st.selectbox("Weather", ["Good","Bad"], key=f"we{i}")

        deliveries.append([d, w, p, we])

# ================================
# INPUT: BATTERY
# ================================
st.header("🔋 Drone Battery Status")

col1, col2 = st.columns(2)

with col1:
    battery_A = st.slider("Drone A Battery (%)", 10, 100, 80)

with col2:
    battery_B = st.slider("Drone B Battery (%)", 10, 100, 80)

# ================================
# OPTIMIZATION BUTTON
# ================================
if st.button("🚀 Optimize Delivery Plan"):

    # Convert data
    input_data = []
    for d in deliveries:
        dist, wt, pr, we = d
        pr_val = {"Low":1,"Medium":2,"High":3}[pr]
        we_val = {"Good":0,"Bad":1}[we]
        input_data.append([dist, wt, pr_val, we_val])

    input_array = np.array(input_data)
    clusters = kmeans.predict(input_array)

    # Prepare delivery list
    delivery_list = []
    for i, delivery in enumerate(input_array):
        delivery_list.append({
            "id": i+1,
            "distance": delivery[0],
            "weight": delivery[1],
            "priority": delivery[2],
            "weather": delivery[3],
            "cluster": clusters[i]
        })

    # Sort by priority
    delivery_list = sorted(delivery_list, key=lambda x: -x["priority"])

    # ================================
    # DRONES
    # ================================
    drones = {
        "Drone A": {"capacity": 10, "load": 0, "deliveries": [], "time": 0, "cost": 0, "battery": battery_A},
        "Drone B": {"capacity": 10, "load": 0, "deliveries": [], "time": 0, "cost": 0, "battery": battery_B}
    }

    # ================================
    # ASSIGNMENT
    # ================================
    for d in delivery_list:

        drone = min(drones, key=lambda x: drones[x]["load"])

        weather_factor = 1.2 if d["weather"] == 1 else 1.0
        battery_needed = d["distance"] * (1 + d["weight"]/10) * weather_factor

        if (drones[drone]["load"] + d["weight"] <= drones[drone]["capacity"]
            and drones[drone]["battery"] >= battery_needed):

            drones[drone]["load"] += d["weight"]
            drones[drone]["battery"] -= battery_needed
            drones[drone]["deliveries"].append(d)

            # Time prediction
            pred_time = rf.predict([[d["distance"], d["weight"], d["priority"], d["weather"]]])[0]
            drones[drone]["time"] += pred_time

            # Cost
            fuel_cost = d["distance"] * 5
            weight_cost = d["weight"] * 2
            weather_penalty = 10 if d["weather"] == 1 else 0

            drones[drone]["cost"] += (fuel_cost + weight_cost + weather_penalty)

        else:
            st.warning(f"🔋 {drone} cannot take Delivery #{d['id']} → Needs charging")

    # ================================
    # DISPLAY PLAN
    # ================================
    st.header("🚁 Final Delivery Plan")

    col1, col2 = st.columns(2)

    for i, (drone, data) in enumerate(drones.items()):
        with (col1 if i == 0 else col2):

            st.subheader(f"🚁 {drone}")

            if data["deliveries"]:

                ordered = sorted(
                    data["deliveries"],
                    key=lambda x: (-x["priority"], x["distance"])
                )

                st.success(f"{drone} will handle deliveries: {[d['id'] for d in ordered]}")

                # ================================
                # ROUTE TIMELINE
                # ================================
                st.markdown("### 🛣 Route Timeline")

                current_time = 0

                for i, d in enumerate(ordered):
                    pred_time = rf.predict([[d["distance"], d["weight"], d["priority"], d["weather"]]])[0]

                    st.write(
                        f"Step {i+1} → Delivery #{d['id']} | ETA: {round(current_time + pred_time,2)} min"
                    )

                    current_time += pred_time

                st.write(f"📦 Load: {round(data['load'],2)} kg")
                st.write(f"⏱ Time: {round(data['time'],2)} min")
                st.write(f"💰 Cost: ₹{round(data['cost'],2)}")
                st.write(f"🔋 Remaining Battery: {round(data['battery'],2)}%")

                # ================================
                # BATTERY DECISION
                # ================================
                st.subheader("🔋 Battery Decision")

                required_battery = 0

                for d in ordered:
                    wf = 1.2 if d["weather"] == 1 else 1.0
                    required_battery += d["distance"] * (1 + d["weight"]/10) * wf * 2

                if data["battery"] >= required_battery:
                    st.success("✅ You are good to go and return safely")

                elif data["battery"] >= required_battery * 0.6:
                    st.warning("⚠️ Battery may not last full trip")

                    farthest = max(ordered, key=lambda x: x["distance"])
                    station = nearest_station(farthest["distance"])

                    st.info(
                        f"🔌 Suggested charging stop: {station['name']} (~{station['distance']} km)"
                    )

                else:
                    st.error("❌ Not enough battery → Charge before dispatch")

            else:
                st.warning("No deliveries assigned")

    # ================================
    # GLOBAL SUMMARY
    # ================================
    total_time = sum(d["time"] for d in drones.values())
    total_cost = sum(d["cost"] for d in drones.values())

    st.header("📊 Overall Summary")

    st.write(f"Total Deliveries: {num_deliveries}")
    st.write(f"Total Time: {round(total_time,2)} min")
    st.write(f"Total Cost: ₹{round(total_cost,2)}")

    st.info("""
This system uses AI to:
- Group deliveries intelligently
- Assign drones efficiently
- Optimize time and cost
- Ensure battery feasibility
- Suggest charging when required
""")