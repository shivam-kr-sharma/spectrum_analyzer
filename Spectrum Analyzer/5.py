# For fitting the model from colab.

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv

class GaussianBesselModel:
    def __init__(self, ac, beta, wc, omega, sigma):
        self.ac = ac
        self.beta = beta
        self.wc = wc
        self.omega = omega
        self.sigma = sigma

    def gaussian(self, x, mu, sigma):
        return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

    def calculate_function_values(self, ws, k_max):
        func_values = []

        for w in ws:
            summation1 = 0
            summation2 = 0
            for k in range(1, k_max + 1):
                summation1 += jv(k, self.beta) ** 2 * self.gaussian(w, self.wc + k * self.omega, self.sigma)
                summation2 += jv(k, self.beta) ** 2 * self.gaussian(w, self.wc - k * self.omega, self.sigma)
            function = self.ac * (jv(0, self.beta) ** 2 * self.gaussian(w, self.wc, self.sigma) + summation1 + summation2)
            func_values.append(function)

        return func_values
    


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
powers_linear = (voltages**2) # In this fitting process I need only the values of beta, which is causing the relative power shifts of the carrier and the sidebands. So I don't care what the actual power of the carrier frequency is.


# Find the frequency corresponding to the maximum power
max_power_index = np.argmax(powers_linear)
frequency_max_power = frequencies[max_power_index]

print(f'Frequency corresponding to the maximum power: {frequency_max_power:.2f} Hz')

# Define the theoretical model
def power_model(frequency, P0, beta, f_c):
    n = (frequency - f_c) / 6e6  # Assuming 6 MHz modulation frequency
    return P0 * jn(n, beta)**2

# Initial guess for the parameters: P0, beta, and carrier frequency f_c
initial_guess = [max(powers_linear), 1.0, frequency_max_power, 6e6, 1e6/2.35]

# Perform the curve fitting
popt, pcov = curve_fit(GaussianBesselModel().calculate_function_values, frequencies, powers_linear, p0=initial_guess)

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
