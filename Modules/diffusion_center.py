"""
just try diffusion process with center coordinate and see what happened
"""

import csv
from torch.utils.data import Dataset, DataLoader

import torch
import matplotlib.pyplot as plt
import torch.nn as nn

import math
from tqdm import tqdm
import copy

class Data_puzzle(Dataset):
    def __init__(self, csv_file, dataset, enlarge_factor=100, delimiter='\t', scale=1, offset=0):
        # repeat each point by a enlarge factor
        self.enlarge_factor = enlarge_factor
        self.points = []
        with open(csv_file, newline='') as f:
            for name, *rest in csv.reader(f, delimiter=delimiter):
                if name == dataset:
                    # cast to float
                    point = torch.tensor(list(map(float, rest)))
                    # rescale
                    self.points.append((point - offset) / scale)

    def __len__(self):
        # enrlarge the effective size
        return len(self.points) * self.enlarge_factor

    def __getitem__(self, i):
        # repeated points
        return self.points[i % len(self.points)]

def plot_batch(batch):
    batch = batch.cpu().numpy()
    plt.scatter(batch[:,0], batch[:,1], marker='.')
    plt.show()


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time, device):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim )
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class MLP_6L(nn.Module):
    def __init__(self, input_size, output_size, dropout_prob = 0.5):
        hidden_sizes = [16, 128, 256, 128, 16]
        super(MLP_6L, self).__init__()
        self.dropout_prob = dropout_prob
        self.gelu = nn.GELU()
        # input layer to hidden layer
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        # hidden layers
        self.hidden_layers = nn.ModuleList()
        # create hidden layer to respect hidden dimension chosen
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                self.gelu,
                nn.Dropout(p = self.dropout_prob)
            ))
        # Define output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.positional_encoding = SinusoidalPositionEmbeddings(2)

    def forward(self, x, t, device):
        # find positional embedding of t
        pos_e = self.positional_encoding.forward(t,device)
        # concatenate to noise position
        x = torch.cat([x, pos_e], dim=1)
        x = self.gelu(self.fc1(x))
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


def train(model, batchsize, device, path):
    # ../../Data/csv/same_centroids_only.tsv
    dataset = Data_puzzle(csv_file=path , dataset='0')
    loader = DataLoader(dataset, batch_size=batchsize)
    # train loop
    LR = 1e-3
    diffusion = diffusionModel.DiffusionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    EPOCH = 1500

    for epoch in tqdm(range(EPOCH)):
        for batch in loader:
            optimizer.zero_grad()
            # choose the timesteps for each element of the batch
            t = torch.randint(0, diffusion.timesteps, (batch.shape[0],)).long().to(device)
            # apply forward of diffusion model to obtain noisy points
            batch_noisy, noise = diffusion.forward(batch, t, device)
            # use the model to estimate the noise
            predicted_noise = model(batch_noisy, t)
            # loss
            loss = torch.nn.functional.mse_loss(noise, predicted_noise)
            loss.backward()
            optimizer.step()


def save_model(model, model_name, path):
   torch.save({
                'model_state_dict': model.state_dict(),
            }, path + model_name + ".pth.zip")

def load_model(model, model_name):
  path = "/content/csv_puzzle/model/" + model_name + ".pth.zip"
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.load_state_dict(torch.load(path, map_location=torch.device(device))['model_state_dict'])
  return model

if __name__ == "__main__":

    dataset = Data_puzzle(csv_file='../../Data/csv/same_centroids_only.tsv', dataset='0',enlarge_factor=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(dataset, batch_size=400)
    # train loop
    model = MLP_6L(4,2)
    train(model, 400, device, "../../Data/csv/same_centroids_only.tsv")





