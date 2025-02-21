# Import necessary libraries
import torch
import numpy as np


# Create a tensor filled with zeros of size (3,3)
tensor_zeros = torch.zeros((3,3))
print('Tensor of zeros:\n', tensor_zeros)


# Create a random tensor of size (3,3)
tensor_random = torch.rand((3,3))
print('Random Tensor:\n', tensor_random)


# DON'T WRITE HERE

# Perform basic tensor addition
tensor_sum = tensor_zeros + tensor_random
print('Sum of tensors:\n', tensor_sum)


# DON'T WRITE HERE

# Check if CUDA is available and create a tensor on GPU if possible
if torch.cuda.is_available():
    tensor_gpu = torch.rand((3,3)).to('cuda')
    print('Tensor on GPU:\n', tensor_gpu)
else:
    print('CUDA is not available')


# DON'T WRITE HERE

# Convert PyTorch tensor to NumPy array
numpy_array = tensor_random.numpy()
print('Converted NumPy array:\n', numpy_array)


# DON'T WRITE HERE

# Reshape a tensor
tensor_reshaped = tensor_random.view(9)
print('Reshaped tensor:\n', tensor_reshaped)


# DON'T WRITE HERE

# Gradient tracking example
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
out.backward()
print('Gradient of x:\n', x.grad)


# DON'T WRITE HERE

# Additional code section

# DON'T WRITE HERE

# Additional code section

# DON'T WRITE HERE


# Step 9: Create a tensor "y" that can be matrix-multiplied with "x"
import torch

# Assuming x has shape (3, 2)
x = torch.tensor([[1, 2],
                  [3, 4],
                  [5, 6]])

# Create tensor y with shape (2, 3) to allow matrix multiplication
y = torch.tensor([[2, 2, 1],
                  [4, 1, 0]])

print("Tensor y:")
print(y)

# Step 10: Find the matrix product of x and y
result = torch.matmul(x, y)
print("Matrix product of x and y:")
print(result)
