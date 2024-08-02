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

plt.figure(figsize=(10, 6))
plt.plot(frequencies, powers_linear, 'bo', label='Experimental Data')
# plt.plot(frequencies_fit, 10 * np.log10(powers_fit / 1e-3), 'r-', label=f'Theoretical Fit (β = {beta_fit:.2f})')
plt.xlabel('Frequency (Hz)')
plt.ylabel('voltages')
plt.title('Frequency Modulated Laser Sidebands')
# plt.legend()
# plt.grid(True)
plt.show()