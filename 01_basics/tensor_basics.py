
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
