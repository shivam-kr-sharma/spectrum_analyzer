# This code can do the fit with the experimental data and tell you the beta.
# Power is normalized.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.optimize import curve_fit

class GaussianBesselModel:
    def __init__(self, beta, wc):
        self.beta = beta
        self.wc = wc

    def gaussian(self, x, mu):
        return np.exp(-((x - mu) ** 2) / (2 * (1.9e6/2.35) ** 2))

    def bessel_fit(self, ws):
        func_values = []
        for w in ws:
            summation1 = 0 
            summation2 = 0
            for k in range(1, 21):
                summation1 += (jv(k, self.beta) ** 2) * self.gaussian(w, self.wc + k * (6e6))
                summation2 += (jv(k, self.beta) ** 2) * self.gaussian(w, self.wc - k * (6e6))
            function = jv(0, self.beta) ** 2 * self.gaussian(w, self.wc) + summation1 + summation2
            func_values.append(function)
        return func_values

def model(ws, beta, wc):
    gb_model = GaussianBesselModel(beta, wc)
    return gb_model.bessel_fit(ws)

# Load the experimental data from an CSV file
data = pd.read_csv('Data/30MVPP.CSV')
frequencies = data['[Hz]'].values
voltages = data['Trace1[V]'].values
power = voltages ** 2

# Normalize the power data
power = power / np.max(power)

# Initial guess for beta and wc
initial_guess = [1.0, frequencies[np.argmax(power)]]  # Start with beta=1 and peak frequency as wc

# Fit the model to the data
params, params_covariance = curve_fit(model, frequencies, power, p0=initial_guess)

# Extract fitted parameters
beta_opt, wc_opt = params
print(f"Optimal beta: {beta_opt}")
print(f"Optimal wc: {wc_opt}")

# Get the fitted model data
fitted_model = model(frequencies, beta_opt, wc_opt)
# Normalize the fitted model data
fitted_model = np.array(fitted_model) / np.max(fitted_model)

# Plot the experimental data and the fitted model
plt.figure(figsize=(10, 6))
plt.plot(frequencies, power, label='Experimental Data', marker='o', linestyle='')
plt.plot(frequencies, fitted_model, label='Fitted Model', linestyle='-')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Normalized Power (V^2)')
plt.legend()
plt.show()
