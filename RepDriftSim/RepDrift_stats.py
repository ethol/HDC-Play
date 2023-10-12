import numpy as np
from scipy import spatial
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from matplotlib import pyplot as plt
import cmath

pi = np.pi


def gaussian_decay(x, sigma):
    return np.exp(-1 * (x / sigma) ** 2)


def complex_cosine_similarity(u, v):
    # Compute the dot product of u and v using numpy's dot function
    dot_product = np.dot(u, np.conjugate(v))

    # Calculate the magnitudes of u and v using numpy's linalg.norm function
    magnitude_u = np.linalg.norm(u)
    magnitude_v = np.linalg.norm(v)

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

    decay = gaussian_decay(t, s)
    new_vector = vector.copy()



    # for i in range(t):
        # drifting_vector = np.random.normal(0, 5, size=D) * 2 * pi
    drifting_vector = np.random.random(D) * 2 * pi

    # drift_rate = np.clip(np.abs(np.random.normal(0.8, 0.25, size=D)), 0, 1)
    # drift_rate = drift_rate * (1 - decay)
    # drift_reverse_rate = 1 - drift_rate

    # drifted_vector = (new_vector * drift_reverse_rate) + (drifting_vector * drift_rate)
    drifted_vector = (new_vector * decay) \
                     # + (drifting_vector * (1-decay))
    new_vector = drifted_vector

    complex_vector_drift_attractor = const_complex * new_vector
    complex_vector_drift_attractor = np.exp(complex_vector_drift_attractor)

    return complex_cosine_similarity(complex_vector, complex_vector_drift_attractor)
    # return spatial.distance.cosine(complex_vector, complex_vector_drift_attractor)
    # return spearmanr(complex_vector, complex_vector_drift_attractor).correlation
    # return pearsonr(complex_vector, complex_vector_drift_attractor)[0]


D = 250
sigma = 5
samples = 1000

avg = []


# Intrepid the angel as a spike
for t in range(-20, 20):
    dist = []
    # avg.append(gaussian_decay(t, sigma))
    for j in range(samples):
        dist.append(exp(D, sigma, t))
    print("dist", dist)
    avg.append(np.mean(dist))
#
# angles_radians = [cmath.phase(z) for z in avg]
# angles_degrees = [angle * 180 / cmath.pi for angle in angles_radians]
# print(angles_degrees)

fig, ax = plt.subplots(3)
ax[0].bar(
    np.arange(len(avg)),
    np.imag(avg)
    # ax=ax,
    # kde=False,
    # stat="density",
    # color=[0.863, 0.1835, 0.1835, 0.5],
    # common_norm=False
)

ax[1].bar(
    np.arange(len(avg)),
    np.real(avg)
    # ax=ax,
    # kde=False,
    # stat="density",
    # color=[0.863, 0.1835, 0.1835, 0.5],
    # common_norm=False
)

ax[2].bar(
    np.arange(len(avg)),
    np.sum([np.real(avg), np.imag(avg)], axis=0)
    # ax=ax,
    # kde=False,
    # stat="density",
    # color=[0.863, 0.1835, 0.1835, 0.5],
    # common_norm=False
)

print(avg)
plt.show()
