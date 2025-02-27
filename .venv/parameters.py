import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd  # Import pandas for CSV handling

# Define the parameters according to the small range (R3)
# Time range (tp): 1–10 s
start_time = 1
end_time = 10
delta_t = 0.05  # Adjusted for practicality

tp = np.arange(start_time, end_time, delta_t)
print(f"Length of tp: {len(tp)}")
print("First five values:", tp[:5])
print("Last five values:", tp[-5:])
print(tp)
print(type(tp))

# Define V_max (randomly select from 1 to 10 V as per R3)
V_max = random.uniform(1, 10)  # Random V_max between 1 and 10 V
print(f"Selected V_max: {V_max:.2f} V")

# Calculate the time for one full cycle (0 to V_max to -V_max to 0)
# Time to reach V_max from 0: t_max = V_max / 5 (sweep rate = 5 V/s)
t_max = V_max / 5
cycle_period = 3 * t_max  # Full cycle duration (one complete triangle)

# Normalize time to fit within 1–10 s and create a periodic triangular wave
# Shift time to start at 0 for simplicity, then adjust back to 1–10 s
t_normalized = tp - 1  # Shift to start at 0 s
t_periodic = t_normalized % cycle_period  # Create periodic behavior within the cycle

# Formulate voltage as a triangular wave with sweep rate of 5 V/s
V_t = np.zeros_like(tp, dtype=float)
for i, t in enumerate(t_periodic):
    if t < t_max:  # Rising phase (0 to V_max)
        V_t[i] = 5 * t
    elif t < 2 * t_max:  # Falling phase (V_max to -V_max)
        V_t[i] = V_max - 5 * (t - t_max)
    else:  # Returning phase (-V_max to 0)
        V_t[i] = -V_max + 5 * (t - 2 * t_max)

# Ensure voltage stays within reasonable bounds (e.g., -V_max to V_max)
V_t = np.clip(V_t, -V_max, V_max).round(2)  # Round to 2 decimal places for readability

# Plot the voltage vs. time
plt.figure(figsize=(10, 6))
plt.plot(tp, V_t, 'b-', label=f'Voltage (V_max = {V_max:.2f} V)')
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Voltage Sweep Over Time (Triangular Wave, 5 V/s)')
plt.legend()

# Save the plot in the same directory as the script
plt.savefig('voltage_vs_time.png', dpi=300, bbox_inches='tight')  # Saves as PNG with high resolution
plt.show()

# Print the first few values of V_t to verify
print("First five voltage values:", V_t[:5])
print("Last five voltage values:", V_t[-5:])

# Define time-independent parameters for one device configuration (R3 range)
# Randomly select values within the R3 ranges (as in your tables)
h = random.uniform(1, 2)  # Geometry, 1–2 μm
C = 10 ** random.uniform(19, 21)  # Doping, 10^19–10^21 m^(-3)
delta = random.uniform(0.28, 0.32)  # Vacancy energy, 0.28–0.32 eV
mu_n = 10 ** np.random.uniform(-4, -3.39794)  # Electron mobility, 10^(-4)–4×10^(-4) m^2/(Vs)
mu_p = 10 ** np.random.uniform(-4, -3.39794)  # Hole mobility, same range as mu_n for simplicity
mu_x = 10 ** np.random.uniform(-15, -13)  # Vacancy mobility, 10^(-15)–10^(-13) m^2/(Vs)
# Ensure mu_x is not zero or extremely small by checking and adjusting if necessary
if mu_x == 0.0 or mu_x < 1e-15:
    mu_x = 1e-15  # Set a minimum value within the range
phi_L = random.uniform(0.05, 0.17)  # Left Schottky barrier, 0.05–0.17 eV
phi_R = random.uniform(0.05, 0.17)  # Right Schottky barrier, 0.05–0.17 eV

# Print the selected time-independent parameters with high precision for mu_x
print("\nTime-independent parameters for this configuration:")
print(f"h (μm): {h:.2f}")
print(f"C (m^-3): {int(C):,}")  # Format C as integer without scientific notation
print(f"δ (eV): {delta:.2f}")
print(f"μ_n (m^2/(Vs)): {mu_n:.6f}")
print(f"μ_p (m^2/(Vs)): {mu_p:.6f}")
print(f"μ_x (m^2/(Vs)): {mu_x:.15f}")  # Show high precision for mu_x
print(f"φ_L (eV): {phi_L:.2f}")
print(f"φ_R (eV): {phi_R:.2f}")

# Save time (tp), voltage (V_t), and time-independent parameters to a CSV file (without current)
data = {
    'Time (s)': np.tile(tp, 1),
    'Voltage (V)': np.tile(V_t, 1),
    'h (μm)': np.full(len(tp), h).round(2).astype(str),  # Format as string with 2 decimals
    'C (m^-3)': np.full(len(tp), int(C)).astype(str),  # Format as integer string
    'δ (eV)': np.full(len(tp), delta).round(2).astype(str),  # Format as string with 2 decimals
    'μ_n (m^2/(Vs))': np.full(len(tp), mu_n).round(6).astype(str),  # Format as string with 6 decimals
    'μ_p (m^2/(Vs))': np.full(len(tp), mu_p).round(6).astype(str),
    'μ_x (m^2/(Vs))': np.full(len(tp), mu_x).round(15).astype(str),  # Format with high precision
    'φ_L (eV)': np.full(len(tp), phi_L).round(2).astype(str),
    'φ_R (eV)': np.full(len(tp), phi_R).round(2).astype(str)
}
df = pd.DataFrame(data)
df.to_csv('device_input_data.csv', index=False, encoding='utf-8-sig')  # Use UTF-8 with BOM for Excel compatibility

print("\nData has been saved to 'device_input_data.csv'")

# Read the CSV file
df = pd.read_csv('../../../../Desktop/device_input_data.csv')

# Display the first few rows (default is 5 rows)
print("\nFirst few rows of the CSV:")
print(df.head())

# Or display all rows if the file is small
print("\nFull CSV content:")
print(df)