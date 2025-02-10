import torch

# Create tensor with shape (1, 3, 1, 4)
t = torch.rand(1, 3, 1, 4)

# Apply t.squeeze()
squeezed_t = t.squeeze()

# Option 1: t.squeeze(2)
option1 = t.squeeze(2)

# Option 2: t.squeeze(0)
option2 = t.squeeze(0)

# Option 3: t.reshape(3, 4)
try:
    option3 = t.reshape(3, 4)
except RuntimeError as e:
    option3 = str(e)

# Print shapes
print("Original shape:", t.shape)
print("t.squeeze():", squeezed_t.shape)
print("t.squeeze(2):", option1.shape)
print("t.squeeze(0):", option2.shape)
print("t.reshape(3, 4):", option3.shape)
