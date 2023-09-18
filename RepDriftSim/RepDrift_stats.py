import numpy as np
from scipy import spatial
import seaborn as sns
from matplotlib import pyplot as plt
import cmath

pi = np.pi


def gaussian_decay(x, sigma):
    return np.exp(-1 * (x / sigma) ** 2)


def complex_cosine_similarity(u, v):
    # Compute the dot product of u and v
    dot_product = sum(u_i * v_i.conjugate() for u_i, v_i in zip(u, v))

    # Calculate the magnitudes of u and v
    magnitude_u = cmath.sqrt(sum(abs(u_i) ** 2 for u_i in u))
    magnitude_v = cmath.sqrt(sum(abs(v_i) ** 2 for v_i in v))

    # Calculate the cosine similarity
    if magnitude_u != 0 and magnitude_v != 0:
        similarity = dot_product / (magnitude_u * magnitude_v)
    else:
        similarity = 0.0  # Handle division by zero

    return similarity

def exp(D, s, t):
    # Generate a random vector in the complex plane
    vector = np.random.random(D) * 2 * pi
    const_complex = np.array([0 + 1j])
    complex_vector = const_complex * vector
    complex_vector = np.exp(complex_vector)

    drifting_vector = np.random.normal(0, 1, size=D) * 2 * pi
    # drifting_vector = np.random.random(D) * 2 * pi

    decay = gaussian_decay(t, s)

    drifted_vector = (vector * decay) + (drifting_vector * (1 - decay))

    complex_vector_drift = const_complex * drifted_vector
    complex_vector_drift = np.exp(complex_vector_drift)

    return complex_cosine_similarity(complex_vector, complex_vector_drift)


D = 1000
sigma = 20
samples = 1000

avg = []
for t in range(0, 20):
    dist = []
    # avg.append(gaussian_decay(t, sigma))
    for j in range(samples):
        dist.append(exp(D, sigma, t))
    print("dist", dist)
    avg.append(np.mean(dist))

# fig, ax = plt.subplots()
plt.bar(
    np.arange(len(avg)),
    avg
    # ax=ax,
    # kde=False,
    # stat="density",
    # color=[0.863, 0.1835, 0.1835, 0.5],
    # common_norm=False
)

print(avg)
plt.show()
