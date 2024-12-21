#!/bin/env python3
"""Follow tensor tutorial from https://pytorch.org/tutorials/beginner/basics/tensor_tutorial.html."""

import datetime

print(f"{datetime.datetime.now(datetime.timezone.utc)} Importing numpy")

import numpy as np  # noqa: E402

print(f"{datetime.datetime.now(datetime.timezone.utc)} Importing torch")
import torch  # noqa: E402

print(f"{datetime.datetime.now(datetime.timezone.utc)} Import done, creating tensors")

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
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

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

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# tensor attributes
print(f"Shape of tensor: {rand_tensor.shape}")
print(f"Datatype of tensor: {rand_tensor.dtype}")
print(f"Device tensor is stored on: {rand_tensor.device}")

# copy tensor to GPU, on AMD / ROCm systems this will also return true
if torch.cuda.is_available():
    rand_tensor = rand_tensor.to("cuda")
    print(f"Device tensor is stored on: {rand_tensor.device}")
