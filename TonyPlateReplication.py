import numpy as np
from scipy import spatial

# dimentionality
n = 512
# number of vectors
m = 1000
# size of bundle
K = 4
# number of trails
T = 1000

# https://colab.research.google.com/github/wilkieolin/VSA-notebooks/blob/main/VSA_Introduction_2_Bundling.ipynb
def bundle(*symbols):
    symbols = np.stack(symbols, axis=0)
    # convert each angle to a complex number
    pi = np.pi
    j = np.array([0 + 1j])

    # sum the complex numbers to find the bundled vector
    cmpx = np.exp(pi * j * symbols)
    bundle = np.sum(cmpx, axis=0)
    # convert the complex sum back to an angle
    bundle = np.angle(bundle) / pi
    bundle = np.reshape(bundle, (1, -1))
    return bundle


def random_vectors(vectors, number_of_vectors):
    bundle_index = np.random.choice(np.arange(m), number_of_vectors, replace=False)
    rand_vec = []
    for i in bundle_index:
        rand_vec.append(vectors[i])
    return rand_vec, bundle_index


def create_test():
    vectors = np.random.random((m, n)) * 2 - 1

    to_bundle, bundle_index = random_vectors(vectors, K)
    bun = bundle(*to_bundle)

    pos = []
    neg = []

    for i in range(0, m):
        # dis = 1 - spatial.distance.cosine(bun, vectors[i])
        dis = np.dot(bun, vectors[i])
        if i in bundle_index:
            pos.append(dis)
        else:
            neg.append(dis)
    return max(neg) < min(pos)

succ = 0
print("")
for i in range(0, T):
    if i % (T/100) == 0:
        print(f"\rprogress {i / T}", end=" ", flush=True)
    if create_test():
        succ += 1

print("successes", succ / T)
