import random
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

class ANet(nn.Module):
    def __init__(self, learning_rate, layer_sizes, hidden_activation_function, optimizer):
        super(ANet, self).__init__()
        layer_sizes = [[layer_sizes[i], layer_sizes[i+1]] for i in range(len(layer_sizes)-1)]
        layers = []
        for layer_size in layer_sizes:
            layers.append(nn.Linear(layer_size[0], layer_size[1]))
            layers.append(hidden_activation_function())
        layers.append(nn.Softmax(dim=1))
        self.layers = nn.Sequential(*layers)
        self.criterion = nn.MSELoss()
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)

    def forward(self, inputs, masks):
        output_tensor = self.layers(torch.FloatTensor(inputs))
        return f.normalize(output_tensor * torch.FloatTensor(masks), p=1, dim=1)

    def do_training(self, rbuf, batch_size=128):
        self.train()
        inputs = []
        masks = []
        targets = []
        for case in random.sample(rbuf, min(batch_size, len(rbuf))):
            inputs.append(case[0])
            masks.append(case[1]) 
            targets.append(case[2])
        outputs = self(inputs, masks)
        # Train
        self.optimizer.zero_grad()
        loss = self.criterion(outputs, torch.FloatTensor(targets))
        loss.backward()
        self.optimizer.step()
        self.eval()
        
    def save(self, path):
        torch.save(self, path)

# After result is produced, set all non available moves to 0, then normalize the rest.