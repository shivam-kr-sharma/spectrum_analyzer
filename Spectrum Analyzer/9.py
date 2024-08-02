import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.optimize import curve_fit

class GaussianBesselModel:
    def __init__(self, beta, wc, amplitude):
        self.beta = beta
        self.wc = wc
        self.amplitude = amplitude

    def gaussian(self, x, mu):
        return np.exp(-((x - mu) ** 2) / (2 * (1e6/2.35) ** 2))/(1e6/2.35)*np.sqrt(2*np.pi)

    def bessel_fit(self, ws):
        func_values = []
        for w in ws:
            summation1 = 0 
            summation2 = 0
            for k in range(1, 21):
                summation1 += (jv(k, self.beta) ) * self.gaussian(w, self.wc + k * (6e6)) # 
                summation2 += (jv(k, self.beta) ) * self.gaussian(w, self.wc - k * (6e6))
            function = self.amplitude*((jv(0, self.beta) )* self.gaussian(w, self.wc) + summation1 + summation2)
            func_values.append(function)
        return func_values

def model(ws, beta, wc, amplitude):
    gb_model = GaussianBesselModel(beta, wc, amplitude)
    return gb_model.bessel_fit(ws)

# Load the experimental data from a CSV file
data = pd.read_csv('Data/0MV.CSV')
frequencies = data['[Hz]'].values
voltages = data['Trace1[V]'].values
power = (voltages**2)/50

# Initial guess for beta, wc, and amplitude
initial_guess = [1.0, frequencies[np.argmax(power)], np.max(power)]  # Start with beta=1, peak frequency as wc, and max power as amplitude

# Fit the model to the data
params, params_covariance = curve_fit(model, frequencies, power, p0=initial_guess)

# Extract fitted parameters
beta_opt, wc_opt, amplitude_opt = params
print(f"Optimal beta: {beta_opt}")
print(f"Optimal wc: {wc_opt}")
print(f"Optimal amplitude: {amplitude_opt}")

# Plot the experimental data and the fitted model
plt.figure(figsize=(10, 6))
plt.plot(frequencies, power, label='Experimental Data', marker='o', linestyle='')
plt.plot(frequencies, model(frequencies, beta_opt, wc_opt, amplitude_opt), label='Fitted Model', linestyle='-')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (V^2)')
plt.legend()
plt.show()
