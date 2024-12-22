#!/bin/env python3
"""Follow tensor tutorial from https://pytorch.org/tutorials/beginner/basics/tensor_tutorial.html."""

import datetime

print(f"{datetime.datetime.now(datetime.timezone.utc)} Importing numpy")

import numpy as np  # noqa: E402

print(f"{datetime.datetime.now(datetime.timezone.utc)} Importing torch")
import torch  # noqa: E402

print(f"{datetime.datetime.now(datetime.timezone.utc)} Import done, creating tensors\n")

# create tensor from raw data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)

# create tensor from numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

# from another tensor
x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor:\n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random Tensor:\n {x_rand} \n")

# tensor creation with tuple
shape = (
    2,
    3,
    # 4,
    # 5,
)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"\nRandom Tensor:\n {rand_tensor} \n")
print(f"Ones Tensor:\n {ones_tensor} \n")
print(f"Zeros Tensor:\n {zeros_tensor}\n")

# tensor attributes
print(f"Shape of tensor: {rand_tensor.shape}")
print(f"Datatype of tensor: {rand_tensor.dtype}")
print(f"Device tensor is stored on: {rand_tensor.device}\n")

# copy tensor to GPU, on AMD / ROCm systems this will also return true
if torch.cuda.is_available():
    rand_tensor = rand_tensor.to("cuda")
    print(f"Device tensor is stored on: {rand_tensor.device}\n")

print("First row:\n", rand_tensor[0])
print("First column:\n", rand_tensor[:, 0])
print("Last column:\n", rand_tensor[:, -1], "\n")

ones_tensor[:, 1] = 2
print("Modified ones tensor (second column = 2):\n", ones_tensor)
ones_tensor[1] = 3
print("Modified ones tensor (second row = 3):\n", ones_tensor, "\n")

# tensor concatenation
t1 = torch.cat([rand_tensor, ones_tensor, zeros_tensor], dim=1)
print("Concatenated tensor:\n", t1, "\n")

# tensor multiplication
identity = torch.zeros(2, 2)
for i in range(identity.size(dim=1)):
    identity[i, i] = 1
print(f"tensor multiplication (@):\n{rand_tensor @ ones_tensor.T}\n")
print(
    f"tensor multiplication (@) @ identity:\n{rand_tensor @ ones_tensor.T @ identity}\n",
)
print(f"tesnor multiplication (element wise):\n{rand_tensor * zeros_tensor}")
print(f"tesnor multiplication (element wise):\n{rand_tensor * ones_tensor}\n")

# tensor aggregration
print("tensor sum: ", rand_tensor.sum())
print("tensor sum: ", ones_tensor.sum(), "\n")
