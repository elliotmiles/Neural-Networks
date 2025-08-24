import numpy as np
import math

def sigmoid(x):
    return 1/(1 + math.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1- sigmoid(x))

def loss(yHat, target):
    return 0.5 * ( (target - yHat)**2 )

def loss_prime(yHat, target):
    return abs(yHat - target)

def forward(w, b, x):
    z = w*x + b
    return sigmoid(z)

def get_z(w, b, x):
    return w*x + b

#dL/dw = dL/dyHat * dyHat/dz * dz/dw = loss_prime * sigmoid_prime * a
def w_update(w, b, x, yHat, rate):
    deriv = loss_prime(yHat, target) * sigmoid_prime(get_z(w, b, x)) * x

    return w - (rate * deriv)

def b_update(w, b, x, yHat, rate,):
    deriv = loss_prime(yHat, target) * sigmoid_prime(get_z(w, b, x))

    return b - (rate * deriv)

target = 0.5
w = 0.1
b = 0.0
error = 1.0

x = float(input("Input: "))

max_iters = 1000
i = 0

while error > 0.00001 and i < max_iters:
    yHat = forward(w, b, x)
    print(f"{i+1}: {yHat:.6f}")
    error = loss(yHat, target)
    w = w_update(w, b, x, yHat, 0.1)
    b = b_update(w, b, x, yHat, 0.1)
    i+=1

print(f"Final prediction: {round(forward(w, b, x), 6)} in {i} iters")
