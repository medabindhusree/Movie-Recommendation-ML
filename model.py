import numpy as np
import pandas as pd
data = pd.read_csv('rating.csv')
data.shape
data.head()
data['userId'].nunique(), data['movieId'].nunique()
user = data['userId'].value_counts().index
len(user)
map = {k:i for i, k in enumerate(user)}
data['userId'] = data['userId'].map(map)
data.head()
data['movieId'].max()
len(data['movieId'].value_counts().index)
mov = data['movieId'].value_counts().index
map = {k:i for i, k in enumerate(mov)}
data['movieId'] = data['movieId'].map(map)
data['movieId'].max()
data.drop('timestamp', axis = 1, inplace = True)
n = 1000
m = 800
from collections import Counter
ucount = Counter(data['userId'])
mcount = Counter(data['movieId'])
userid = [u for u,c in ucount.most_common(n)]
movieid = [u for u, c in mcount.most_common(m)]
len(userid), len(movieid)
newdata = data[data['userId'].isin(userid) & data['movieId'].isin(movieid)].copy()
newdata.shape

newdata.head()

newdata['userId'].max()

newdata['movieId'].max()

user = newdata['userId'].value_counts().index
map = {k:i for i, k in enumerate(user)}
newdata['userId'] = newdata['userId'].map(map)
mov = newdata['movieId'].value_counts().index
map = {k:i for i, k in enumerate(mov)}
newdata['movieId'] = newdata['movieId'].map(map)

from sklearn.utils import shuffle
newdata = shuffle(newdata)
cutoff = int(0.8*len(newdata))
cutoff

train = newdata.iloc[: cutoff, :]
test = newdata.iloc[cutoff: , :]
train.shape, test.shape

u2m = {}
m2u = {}
um2r = {}
count = 0


def umr(data):
    global count
    count += 1

    if count % 100000 == 0:
        print("Processed: %.3f" % float(count / cutoff))

    i = int(data['userId'])
    j = int(data['movieId'])

    if i not in u2m:
        u2m[i] = [j]
    else:
        u2m[i].append(j)

    if j not in m2u:
        m2u[j] = [i]
    else:
        m2u[j].append(i)

    um2r[(i, j)] = data['rating']


train.apply(umr, axis=1)
um2r_test = {}
count = 0


def umrtest(data):
    global count
    count += 1
    if count % 100000 == 0:
        print("Proposed: %.3f" % float(count / len(data)))

    i = int(data['userId'])
    j = int(data['movieId'])

    um2r_test[(i, j)] = data['rating']


test.apply(umrtest, axis=1)
np.max(list(u2m.keys()))

np.max(list(m2u.keys()))

np.max([m for (i, m), r in um2r_test.items()])

K = 10
N = 1000
M = 800
W = np.random.randn(N, K)
b = np.zeros(N)
U = np.random.randn(M, K)
c = np.zeros(M)
mu = np.mean(list(um2r.values()))


def get_loss(d):
    N = len(d)
    sse = 0

    for k, r in d.items():
        i, j = k
        p = W[i].dot(U[j]) + b[i] + c[j] + mu
        sse += (p - r) * (p - r)
    return sse / N


from datetime import datetime

epochs = 30
reg = 0.01
train_losses = []
test_losses = []

for epoch in range(epochs):
    print("Epochs: ", epoch)
    epoch_start = datetime.now()

    t0 = datetime.now()

    for i in range(N):
        matrix = np.eye(K) * reg
        vector = np.zeros(K)
        bi = 0

        for j in u2m[i]:
            r = um2r[(i, j)]

            matrix += np.outer(U[j], U[j])
            vector += (r - b[i] - c[j] - mu) * U[j]
            bi += (r - W[i].dot(U[j]) - c[j] - mu)

        W[i] = np.linalg.solve(matrix, vector)
        b[i] = bi / (len(u2m[i]) + reg)

        if i % (N // 10) == 0:
            print("i: ", i, "N: ", N)

    print("Updated W & b", datetime.now() - t0)

    t0 = datetime.now()

    for j in range(M):
        matrix = np.eye(K) * reg
        vector = np.zeros(K)
        cj = 0

        try:
            for i in m2u[j]:
                r = um2r[(i, j)]

                matrix += np.outer(W[i], W[i])
                vector += (r - b[i] - c[j] - mu) * W[i]
                cj += (r - W[i].dot(U[j]) - b[i] - mu)

            U[j] = np.linalg.solve(matrix, vector)
            c[j] = cj / (len(m2u[j]) + reg)

            if j % (M // 10) == 0:
                print("j:", j, "M: ", M)

        except KeyError:
            pass

    print("Updated U & c: ", datetime.now() - t0)
    print("epoch Duration: ", datetime.now() - epoch_start)

    train_losses.append(get_loss(um2r))
    test_losses.append(get_loss(um2r_test))

    print("train Loss: ", train_losses[-1])
    print("test loss: ", test_losses[-1])

import matplotlib.pyplot as plt
# plot losses
plt.plot(train_losses, label="train loss")
plt.plot(test_losses, label="test loss")
plt.legend()
plt.show()