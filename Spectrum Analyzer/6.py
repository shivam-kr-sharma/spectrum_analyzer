import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.optimize import minimize

class GaussianBesselModel:
    def __init__(self, beta, wc):
        self.beta = beta
        self.wc = wc

    def gaussian(self, x, mu):
        return np.exp(-((x - mu) ** 2) / (2 * (1e6 / 2.35) ** 2))

    def bessel_fit(self, ws):
        func_values = []
        for w in ws:
            summation1 = 0 
            summation2 = 0
            for k in range(1, 21):
                summation1 += jv(k, self.beta) ** 2 * self.gaussian(w, self.wc + k * (6e6))
                summation2 += jv(k, self.beta) ** 2 * self.gaussian(w, self.wc - k * (6e6))
            function = (8.72e-6) * (jv(0, self.beta) ** 2 * self.gaussian(w, self.wc) + summation1 + summation2)
            func_values.append(function)

        return np.array(func_values)

def objective_function(beta, ws, voltages, wc):
    model = GaussianBesselModel(beta, wc)
    theoretical_values = model.bessel_fit(ws)
    return np.sum((voltages - theoretical_values) ** 2)

# Load data from Excel
data = pd.read_csv('Data/1MV.CSV')
frequencies = data['[Hz]'].values
voltages = data['Trace1[V]'].values

# Estimate wc (central carrier frequency)
wc = np.mean(frequencies)

# Initial guess for beta
initial_beta = 1.0

# Optimize beta
result = minimize(objective_function, initial_beta, args=(frequencies, voltages, wc), bounds=[(0, 10)])
optimal_beta = result.x[0]

# Plotting the results
model = GaussianBesselModel(optimal_beta, wc)
theoretical_values = model.bessel_fit(frequencies)

plt.plot(frequencies, voltages, label='Experimental Data')
plt.plot(frequencies, theoretical_values, label='Theoretical Fit')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Voltage (V)')
plt.title(f'Optimal Beta: {optimal_beta:.2f}')
plt.legend()
plt.show()

print(f'Optimal Beta: {optimal_beta:.2f}')
