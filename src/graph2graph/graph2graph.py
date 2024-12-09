import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import torch_geometric
from torch_geometric.nn import MessagePassing, GATConv, global_mean_pool
from torch_geometric.data import Data
import torch_geometric.transforms
from torch_geometric.loader import DataLoader

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear, Sequential, ReLU

class LidarGraphConverter:
    @staticmethod
    def polar_to_cartesian(distances, angles):
        """
        Convert polar coordinates to Cartesian coordinates
        
        Args:
        - distances (torch.Tensor): Distance measurements (normalized)
        - angles (torch.Tensor): Corresponding angles in radians
        
        Returns:
        - torch.Tensor: Cartesian coordinates (x, y)
        """
        x = distances * torch.cos(angles)
        y = distances * torch.sin(angles)
        return torch.stack([x, y], dim=-1)
    
    @staticmethod
    def create_graph_from_lidar(lidar_scan, wind_features):
        """
        Create a graph representation from lidar scan
        
        Args:
        - lidar_scan (torch.Tensor): Normalized lidar scan of shape (N, 360)
        - wind_features (torch.Tensor): Wind features of shape (N, 2)
        
        Returns:
        - torch_geometric.data.Data: Graph representation
        """
        # Filter out non-collision points (value == 1)
        N = lidar_scan.shape[0]
        angles = torch.linspace(0, 2*math.pi, steps=360)
        
        connect = False
        node_features = []
        edge_idx = []
        edge_attr = []

        for i in range(N):
            collision_distances = lidar_scan[i]
            if collision_distances != 1: # building scanned
                posn = LidarGraphConverter.polar_to_cartesian(collision_distances, angles[i])
                node_features.append(posn)
                if connect:
                    start = len(node_features) - 1
                    end = start - 1
                    edge_idx.append([start, end])
                    edge_idx.append([end, start])
                    diff = node_features[start]-node_features[end]
                    edge_attr.append(torch.unsqueeze(torch.linalg.norm(diff, dim=0, ord=2), 0))
                    edge_attr.append(torch.unsqueeze(torch.linalg.norm(diff, dim=0, ord=2), 0))
                else:
                    connect = True
            elif connect:
                connect = False

        graph = Data(
            x=torch.stack(node_features), 
            edge_index=torch.transpose(torch.tensor(edge_idx, dtype=torch.long), 0, 1), 
            edge_attr=torch.stack(edge_attr),
            pos=torch.stack(node_features)
        )

        return graph


class G2GCore(torch.nn.Module):
    def __init__(self, node_feature_dim=2, hidden_dim=64, pred_size=10):
        super(G2GCore, self).__init__()
        # Define GCN layers
        self.pred_size = pred_size

        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # MLP to project the graph-level embedding to the required output dimension
        self.global_pooling = Linear(hidden_dim, 512)  # Compress graph features
        self.mlp = Sequential(
            Linear(512, 1024),
            ReLU(),
            Linear(1024, pred_size*pred_size*2),
        )
        print(pred_size*pred_size*2)

    def forward(self, x, edge_index):
        """
        Parameters:
        x (torch.Tensor): Node features of shape [num_nodes, node_feature_dim].
        edge_index (torch.Tensor): Edge indices in COO format of shape [2, num_edges].
        
        Returns:
        torch.Tensor: A matrix of shape [output_dim, output_dim].
        """
        # Ensure x and edge_index are on the same device
        # print(x.shape)
        # print(edge_index.shape)
        
        # Pass through GCN layers with nonlinearities
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        # Aggregate node features to graph-level embedding (global mean pooling)
        graph_embedding = x.mean(dim=0)  # [hidden_dim]
        
        # Project graph embedding to the required output size
        compressed_embedding = self.global_pooling(graph_embedding)
        output = self.mlp(compressed_embedding)  # [output_dim]
        
        # Reshape to output_dim x output_dim matrix
        return output.view(self.pred_size, self.pred_size, 2)

class Graph2GraphModel(torch.nn.Module):
    def __init__(self, hidden_channels=64, decoder_S=16):
        super().__init__()
        
        self.graph_converter = LidarGraphConverter
        self.core_model = G2GCore()
    
    def forward(self, x):
        """
        Full forward pass of the Graph2Graph model
        
        Args:
        - x (torch.Tensor): Input tensor of shape (N, 362)
        
        Returns:
        - torch.Tensor: Decoded output of shape (N, 2*(S^2))
        """
        # Split input: first 360 for lidar scan, last 2 for wind features
        lidar_scan = x[:, :360]
        wind_features = x[:, 360:]
        
        # Convert to graph representation
        graph = self.graph_converter.create_graph_from_lidar(lidar_scan[0], wind_features)
        
        y = self.core_model(graph.x, graph.edge_index)
        
        return torch.unsqueeze(y, dim=0)


# Example usage and model initialization
def test_model():
    # Create a sample input tensor
    batch_size = 32
    input_tensor = torch.rand(batch_size, 362)
    input_tensor[:, :360] = torch.where(
        torch.rand(batch_size, 360) < 0.25, 
        torch.ones(batch_size, 360), 
        torch.rand(batch_size, 360)
    )
    
    # Initialize the model
    model = Graph2GraphModel(hidden_channels=64, decoder_S=16)
    
    # Forward pass
    output = model(input_tensor)
    
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    test_model()