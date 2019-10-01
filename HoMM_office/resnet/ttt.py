import numpy as np

for epoch in range(100):
    param = 2 / (1 + np.exp(-10 * (epoch) / 100)) - 1
    print(param)