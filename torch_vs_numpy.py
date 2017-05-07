import numpy as np
import torch

np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
print(
    '\nnumpy: ', np_data,
    '\ntorch data: ', torch_data,
    '\ntensor2array: ', tensor2array,
)

# abs
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)
abs_data = torch.abs(tensor)
print(
    '\ndata: ', data,
    '\nabs data: ', abs_data,
)

# sin
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)
numpy_sin_data = np.sin(data)
sin_data = torch.sin(tensor)
print(
    '\ndata: ', data,
    '\nnp sin data: ', numpy_sin_data,
    '\ntorch sin data: ', sin_data,
)
# sin
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)

print(
    '\nmean: ',
    '\nnumpy: ', np.mean(data),
    '\ntorch: ', torch.mean(tensor),
)

# matrix multiply

data = [[1, 2],[3, 4]]
tensor = torch.FloatTensor(data)

print(
    '\nnumpy: ', np.matmul(data, data),
    '\ntorch: ', torch.mm(tensor, tensor),
)

# matrix dot multiply with torch vs np dot computor
data = [[1, 2],[3, 4]]
tensor = torch.FloatTensor(data)
data = np.array(data)

print(
    '\nnumpy: ', data.dot(data),
    '\ntorch: ', tensor.dot(tensor),
)