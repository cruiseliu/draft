import json
from pathlib import Path
import struct

import torch
from torch import nn

torch.manual_seed(0)

class Part(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.fc = nn.Linear(in_size, out_size)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.activ(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.parts = nn.ModuleList([
            Part(20, 100),
            Part(100, 10),
        ])

    def forward(self, x):
        for part in self.parts:
            x = part(x)
        return x

_dtype_to_format = {
    torch.float32: 'f',
    torch.float64: 'd',
    torch.uint8: 'B',
    torch.int8: 'b',
    torch.int16: 'h',
    torch.int32: 'i',
    torch.int64: 'q',
    torch.bool: '?',
}

def _py_name_to_cs_name(py_name):
    parts = py_name.split('.')
    parts = ['part' + parts[1]] + parts[2:]
    return '.'.join(parts)

def dump_model(model, py_name_to_cs_name=None):
    if py_name_to_cs_name is None:
        py_name_to_cs_name = _py_name_to_cs_name

    Path('weights').mkdir(exist_ok=True)
    meta = []

    for name, tensor in model.named_parameters(recurse=True):
        tensor_meta = {
            'name': name,
            'cs_name': py_name_to_cs_name(name),
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype)
        }
        meta.append(tensor_meta)

        tensor_list = tensor.view(-1).tolist()
        fmt = str(len(tensor_list)) + _dtype_to_format[tensor.dtype]
        tensor_bytes = struct.pack(fmt, *tensor_list)
        Path('weights', name).write_bytes(tensor_bytes)

    Path('weights_meta.json').write_text(json.dumps(meta, indent=4))


def main():
    model = Net()
    x = torch.rand(20)
    print(x)
    y = model(x)
    print(y)
    dump_model(model)

main()
