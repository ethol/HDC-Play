import numpy as np
import cmath
import matplotlib.pyplot as plt

# Random seed
np.random.seed(42)

pi = np.pi

# vector size
D = 1000

# sigma
s = 5

# time
t = 20


def gaussian_decay(x, sigma):
    return np.exp(-1 * (x / sigma) ** 2)


# Generate a random vector in the complex plane
vector = np.random.random(D) * 2 * pi
const_complex = np.array([0 + 1j])
complex_vector = const_complex * vector
complex_vector = np.exp(complex_vector)

drifting_vector = np.random.normal(0, 1, size=D) * 2 * pi

decay = gaussian_decay(t, s)

drifted_vector = (vector * decay) + (drifting_vector * (1 - decay))

complex_vector_drift = const_complex * drifted_vector
complex_vector_drift = np.exp(complex_vector_drift)

# Intrepid the angel as a spike
angles_radians = [cmath.phase(z) for z in complex_vector]
angles_degrees = [angle * 180 / cmath.pi for angle in angles_radians]

angles_radians_drift = [cmath.phase(z) for z in complex_vector_drift]
angles_degrees_drift = [angle * 180 / cmath.pi for angle in angles_radians_drift]

sorted_indices = sorted(range(len(angles_degrees)), key=lambda i: angles_degrees[i])
sorted = [angles_degrees[i] for i in sorted_indices]
sorted_drift = [angles_degrees_drift[i] for i in sorted_indices]
# sorted = angles_degrees


fig, ax = plt.subplots()
ax.set_facecolor('blue')

# Create a line graph with just dots for the data points
plt.plot(sorted_drift, np.arange(len(sorted)), '.', markersize=8, markerfacecolor='yellow',
         markeredgecolor='yellow', label='Data Points', linestyle=' ')

# Set plot labels and title
plt.xlabel('spiking time')
plt.ylabel("Neuron #")
plt.title('Line Graph with Dots for Data Points')

# Show the plot
plt.legend()
plt.grid(True)
plt.show()
