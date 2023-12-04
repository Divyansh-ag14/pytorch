
#%%
# Perform standard imports
import torch
import numpy as np

# %%
## Converting NumPy arrays to PyTorch tensors
# A torch.Tensor is a multi-dimensional matrix containing elements of a single data type.<br>
# Calculations between tensors can only happen if the tensors share the same dtype.<br>
# In some cases tensors are used as a replacement for NumPy to use the power of GPUs.

arr = np.array([1,2,3,4,5])
print(arr)
print(arr.dtype)
print(type(arr))
# %%

x = torch.from_numpy(arr)
# Equivalent to x = torch.as_tensor(arr)
print(x)
# %%

# Print the type of data held by the tensor
print(x.dtype)

# %%

# Print the tensor object type
print(type(x))
print(x.type()) # this is more specific!

# %%

arr2 = np.arange(0.,12.).reshape(4,3)
print(arr2)

# %%

x2 = torch.from_numpy(arr2)
print(x2)
print(x2.type())

# %%

# torch.from_numpy()
# torch.as_tensor()
# torch.tensor()

# Using torch.from_numpy()
arr = np.arange(0,5)
t = torch.from_numpy(arr)
print(t)

# %%

arr[2]=77
print(t)

# %%

# Using torch.tensor()
arr = np.arange(0,5)
t = torch.tensor(arr)
print(t)

# %%

arr[2]=77
print(t)

# %%

## Class constructors
# torch.Tensor()
# torch.FloatTensor()
# torch.LongTensor()

# There's a subtle difference between using the factory function torch.tensor(data) and the class constructor torch.Tensor(data)
# The factory function determines the dtype from the incoming data, or from a passed-in dtype argument.
# The class constructor torch.Tensor() is simply an alias for torch.FloatTensor(data). 
# Consider the following:

data = np.array([1,2,3])
a = torch.Tensor(data)  # Equivalent to cc = torch.FloatTensor(data)
print(a, a.type())

# %%

b = torch.tensor(data)
print(b, b.type())

# %%

c = torch.tensor(data, dtype=torch.long)
print(c, c.type())

# %%

## Creating tensors from scratch
# Uninitialized tensors with .empty()
# torch.empty() returns an uninitialized tensor. 
# Essentially a block of memory is allocated according to the size of the tensor, and any values already sitting in the block are returned. 
# This is similar to the behavior of numpy.empty().

x = torch.empty(4, 3)
print(x)

# %%

# Initialized tensors with .zeros() and .ones()

x = torch.zeros(4, 3, dtype=torch.int64)
print(x)

# %%

# Tensors from ranges
x = torch.arange(0,18,2).reshape(3,3)
print(x)

# %%

x = torch.linspace(0,18,12).reshape(3,4)
print(x)

# %%

# Tensors from data
x = torch.tensor([1, 2, 3, 4])
print(x)
print(x.dtype)
print(x.type())

# %%
x = torch.FloatTensor([5,6,7])
print(x)
print(x.dtype)
print(x.type())

# %%
x = torch.tensor([8,9,-3], dtype=torch.int)
print(x)
print(x.dtype)
print(x.type())

# %%

 #change the dtype of an existing tensor with .type()
print('Old:', x.type())

x = x.type(torch.int64)

print('New:', x.type())
# %%
