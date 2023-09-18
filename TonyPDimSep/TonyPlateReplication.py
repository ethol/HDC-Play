import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import spatial
from sklearn.preprocessing import MinMaxScaler

# dimentionality
n = 256
# number of vectors
m = 1000
# size of bundle
K = 4
# number of trails
T = 1000


# bundle


# simple map architecture
# just add the vectors to bundle 
# https://colab.research.google.com/github/wilkieolin/VSA-notebooks/blob/main/VSA_Introduction_2_Bundling.ipynb
def bundle(*symbols):
    # summing
    symbols = np.stack(symbols, axis=0)
    bundle = np.sum(symbols, axis=0)
    min = np.min(bundle)
    max = np.max(bundle)

    bundle_norm = (bundle - min) / (max - min) * 2 - 1

    # complex
    # symbols = np.stack(symbols, axis=0)
    #
    # # convert each angle to a complex number
    # pi = np.pi
    # j = np.array([0 + 1j])
    # cmpx = np.exp(pi * j * symbols)
    #
    # # sum the complex numbers to find the bundled vector
    # bundle = np.sum(cmpx, axis=0)
    #
    # # convert the complex sum back to an angle
    # bundle = np.angle(bundle) / pi
    # bundle = np.reshape(bundle, (1, -1))
    return bundle_norm


def random_vectors(vectors, number_of_vectors):
    bundle_index = np.random.choice(np.arange(m), number_of_vectors, replace=False)
    rand_vec = []
    for i in bundle_index:
        rand_vec.append(vectors[i])
    return rand_vec, bundle_index


all_dis_neg = []
all_dis_pos = []
avg_dis_po_and_neg = []


def create_test():
    # vectors = np.random.randint(2, size=(m, n))
    vectors = np.random.random((m, n)) \
              * 2 - 1

    to_bundle, bundle_index = random_vectors(vectors, K)
    bun = bundle(*to_bundle)

    pos = []
    neg = []

    for i in range(0, m):
        # dis = 1 - spatial.distance.cosine(bun, vectors[i])
        # dis = spatial.distance.euclidean(bun, vectors[i])
        dis = np.dot(bun, vectors[i])
        if i in bundle_index:
            pos.append(dis)
        else:
            neg.append(dis)
    all_dis_neg.append(neg)
    all_dis_pos.append(pos)

    avg_dis_po_and_neg.append(np.mean(pos))
    avg_dis_po_and_neg.append(np.mean(neg))

    return max(neg) < min(pos)


succ = 0
print("")
for i in range(0, T):
    if i % (T / 100) == 0:
        print(f"\rProgress {i / T}", end=" ", flush=True)
    if create_test():
        succ += 1

sep = np.max(avg_dis_po_and_neg) - np.min(avg_dis_po_and_neg)
print("sep", sep)
print("stdev", np.std(avg_dis_po_and_neg / sep))
all_dis_neg = np.array(all_dis_neg).flatten()
all_dis_pos = np.array(all_dis_pos).flatten()

fig, ax = plt.subplots()
sns.histplot(
    all_dis_neg,
    ax=ax,
    # kde=False,
    stat="density",
    color=[0.863, 0.1835, 0.1835, 0.5],
    common_norm=False
)
sns.histplot(
    all_dis_pos,
    ax=ax,
    # kde=False,
    stat="density",
    color=[0.21875, 0.5546875, 0.234375, 0.5],
    common_norm=False
)
#
# plt.hist([all_dis_neg, all_dis_pos]
#          , bins=int(1000 / 10)
#          , density=True
#          , color=[[1.0, 0, 0, 0.5], [0, 1.0, 0, 0.5]], label=["neg", "pos"])
plt.title(f"Dimensions={n}, Number of vectors={m},\n"
          f" Vectors in bundle={K}, trails={T}", fontdict={'fontweight': "bold"})
plt.show()

print("\nSuccesses", succ / T)
