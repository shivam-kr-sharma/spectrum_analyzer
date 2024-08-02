import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import jn

# Load your data
# Assuming your data is in a CSV file with columns 'frequency' and 'power_dbm'
data = pd.read_csv('Data/0MV.CSV')
frequencies = data['Hz'].values
powers_linear = data['Trace1V'].values
print(type(frequencies))
print(frequencies.shape)




# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# from scipy.special import jn

# # Load your data
# data = pd.read_csv('Data/0MV.CSV')

# # Print the column names to check what they are
# print(data.columns)

# # Assuming the correct column names are 'frequency' and 'power_dbm'
# # You should update these to match the actual column names in your CSV file
# frequencies = data['Hz'].values
# powers_linear = data['Trace1V'].values

# print(type(frequencies))









# # Convert power from dBm to linear scale
# # powers_linear = 10**(powers_dbm / 10)

# # Define the theoretical model
# def power_model(frequency, P0, beta, f_c):
#     n = (frequency - f_c) / 6e6  # Assuming 6 MHz modulation frequency
#     return P0 * jn(n, beta)**2

# # Initial guess for the parameters: P0, beta, and carrier frequency f_c
# initial_guess = [max(powers_linear), 1.0, frequencies[powers_linear.index()]]

# # Perform the curve fitting
# popt, pcov = curve_fit(power_model, frequencies, powers_linear, p0=initial_guess)

# # Extract the parameters
# P0_fit, beta_fit, f_c_fit = popt

# # Calculate the fitted curve
# frequencies_fit = np.linspace(min(frequencies), max(frequencies), 1000)
# powers_fit = power_model(frequencies_fit, P0_fit, beta_fit, f_c_fit)

# # Plot the data and the fit
# plt.figure(figsize=(10, 6))
# plt.plot(frequencies, powers_linear, 'bo', label='Experimental Data')
# plt.plot(frequencies_fit, 10 * np.log10(powers_fit), 'r-', label=f'Theoretical Fit (β = {beta_fit:.2f})')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Power (dBm)')
# plt.title('Frequency Modulated Laser Sidebands')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Print the fitted modulation index
# print(f'Fitted Modulation Index (β): {beta_fit:.2f}')
