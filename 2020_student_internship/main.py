import random
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

Ws = [random.random()*4-2 for _ in range(2)]
#Ws = [2, 1]
len_w = 50
batches = 00000   #0 for anim
batch_size = 10
lr = 0.01
momentum = 0.9
points = 11

#Sets len of Ws to len_w
Ws += [0]*(len_w-len(Ws))

#Returns [1, x, x**2, x**3, ...]
def get_x(X):
    Xs = []
    for i in range(len_w):
        Xs.append(X**i)
    return np.array(Xs)

#Computes w[0] + w[1]*x + w[2]*x**2 + w[3]*x**3 + ...
def func(X, W):
    return get_x(X).dot(W)

#Computes the loss
def loss(X, Y):
    return (Y - func(X, w))**2

def dloss(X, Y):
    return Y - func(X, w)

#trains for one batch
sample = 0
old_d = np.zeros(len_w)
def train_batch(batch):
    global w, sample, old_d
    d = np.zeros(len_w)
    sum_loss = 0
    for step in range(batch_size):
        sample += 1
        point = sample % len(x)
        sum_loss += loss(x[point], y[point])
        dl = dloss(x[point], y[point])
        d += dl * get_x(x[point])
    w += lr * (old_d*momentum + d*(1-momentum))
    old_d = (old_d*momentum + d*(1-momentum))

    return 0.5*sum_loss

#Trains and updates the line
fig, ax = plt.subplots()
line = ax.plot([],[],"r")[0]
def train_batch_anim(batch):
    train_batch(batch)
    line.set_data(x, [func(i, w) for i in x])

#Generates the points
x = []
y = []
for position in range(-(points//2), points//2 + points%2):
    position /= (points//2)
    x.append(position)
    y.append(func(position, Ws) + random.random()*2-1)

#Draws the points
plt.plot(x, [func(x_, Ws) for x_ in x], "g")
plt.plot(x, y, "xb")

#Initializes the weights
w = np.random.random(len_w)-0.5
w[0] = 0

#Starts the training
if batches == 0:
    animation = animation.FuncAnimation(fig, train_batch_anim, interval=1)
    plt.show()
else:
    losses = []
    for i in range(batches):
        losses.append(train_batch(i))
    plt.plot(x, [func(x_, w) for x_ in x], "r")
    plt.show()
    plt.plot(losses)
    plt.show()

print(w)
