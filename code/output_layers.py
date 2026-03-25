from typing import List
import torch
import torch.nn as nn


class ComplexLinear(nn.Module):
    def __init__(self, 
        input_dim : int, 
        output_dim : int, 
        hidden : List[int] = [64, 64], 
        activation_fn : nn.Module = nn.ReLU#nn.Tanh
        ):
        #? nn.ReLU()
        super().__init__()

        last_dim = input_dim

        self.net = []
        for dim in hidden:
            self.net.append(nn.Linear(last_dim, dim))
            self.net.append(activation_fn())
            last_dim = dim

        self.net.append(nn.Linear(last_dim, output_dim))
        
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)
        


class IndependentOutputLayer(nn.Module):
    def __init__(self, 
        vocab_sizes : list, 
        latent_size : int,
        ):
        super().__init__()
        
        self.output_layers = nn.ModuleList([
            nn.Linear(latent_size, vocab_size) for vocab_size in vocab_sizes
        ])

    def forward(self, x):
        #outputs = [torch.softmax(layer(x), dim=-1) for layer in self.output_layers]
        outputs = [layer(x) for layer in self.output_layers]
        return outputs
    


class DependentOutputLayer(nn.Module):
    def __init__(self, 
        vocab_sizes : list, 
        latent_size : int,
        complex : bool = True,
        ):
        super().__init__()

        output_fn = ComplexLinear if complex else nn.Linear
        self.independent_output =  output_fn(latent_size, vocab_sizes[0])
        self.output_layers = nn.ModuleList([
            output_fn(latent_size + vocab_sizes[0], vocab_size) for vocab_size in vocab_sizes[1:]
        ])

    def forward(self, x):
        
        independent_output = self.independent_output(x)
        dependent_outputs = [output_layer(torch.cat([x, independent_output.detach()], dim=-1)) for output_layer in self.output_layers]

        return [independent_output] + dependent_outputs