import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import jn

# Constants
Z = 50  # Impedance in ohms, typically 50 ohms

# Load your data
# Assuming your data is in a CSV file with columns 'frequency' and 'voltage'
data = pd.read_csv('Data/0MV.CSV')
frequencies = data['Hz'].values
voltages = data['Trace1V'].values

# Convert voltage to power in linear scale (W)
powers_linear = (voltages**2) / Z


# Find the frequency corresponding to the maximum power
max_power_index = np.argmax(powers_linear)
frequency_max_power = frequencies[max_power_index]

print(f'Frequency corresponding to the maximum power: {frequency_max_power:.2f} Hz')

# Define the theoretical model
def power_model(frequency, P0, beta, f_c):
    n = (frequency - f_c) / 6e6  # Assuming 6 MHz modulation frequency
    return P0 * jn(n, beta)**2

# Initial guess for the parameters: P0, beta, and carrier frequency f_c
initial_guess = [max(powers_linear), 1.0, frequency_max_power]

# Perform the curve fitting
popt, pcov = curve_fit(power_model, frequencies, powers_linear, p0=initial_guess)

# Extract the parameters
P0_fit, beta_fit, f_c_fit = popt

# Calculate the fitted curve
frequencies_fit = np.linspace(min(frequencies), max(frequencies), 1000)
powers_fit = power_model(frequencies_fit, P0_fit, beta_fit, f_c_fit)

# Plot the data and the fit
plt.figure(figsize=(10, 6))
plt.plot(frequencies, powers_linear, 'bo', label='Experimental Data')
plt.plot(frequencies_fit, 10 * np.log10(powers_fit / 1e-3), 'r-', label=f'Theoretical Fit (β = {beta_fit:.2f})')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dBm)')
plt.title('Frequency Modulated Laser Sidebands')
plt.legend()
plt.grid(True)
plt.show()

# Print the fitted modulation index
print(f'Fitted Modulation Index (β): {beta_fit:.2f}')
print(f'Fitted Carrier Frequency (f_c): {f_c_fit:.2f} Hz')
