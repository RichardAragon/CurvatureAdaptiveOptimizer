import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import silhouette_score

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Dataset: Example knowledge graph (extendable to larger datasets like ConceptNet)
data = [
    ("cat", "animal"),
    ("dog", "animal"),
    ("car", "vehicle"),
    ("bicycle", "vehicle"),
    ("animal", "living_being"),
    ("vehicle", "object"),
    ("living_being", "entity"),
    ("object", "entity"),
]

# Create graph
G = nx.DiGraph()
G.add_edges_from(data)
node_list = list(G.nodes)
embedding_dim = 8
node_to_idx = {node: i for i, node in enumerate(node_list)}

# Visualize the initial graph
plt.figure(figsize=(8, 6))
nx.draw_networkx(G, with_labels=True, node_size=1500, font_size=10)
plt.title("Initial Graph")
plt.show()

# Quantum-Inspired Embeddings
class QuantumEmbedding(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.phase = nn.Parameter(torch.rand(num_nodes, embedding_dim))  # Phase term

    def forward(self, x):
        embed = self.embeddings(x)
        phase_shift = torch.cos(self.phase[x])  # Phase encoding
        return embed * phase_shift

# Quantum-Inspired Attention
class QuantumAttention(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.embedding_dim = embedding_dim

    def forward(self, query, key, value):
        # Compute similarity (attention scores)
        attention_scores = torch.matmul(query, key.T)
        attention_probs = self.softmax(attention_scores)
        output = torch.matmul(attention_probs, value)
        return output, attention_probs

# Dynamic Graph Optimization
class QuantumGraphOptimizer(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super().__init__()
        self.embedding = QuantumEmbedding(num_nodes, embedding_dim)
        self.attention = QuantumAttention(embedding_dim)

    def forward(self, query_node, key_nodes):
        query_embed = self.embedding(query_node)
        key_embeds = self.embedding(key_nodes)
        value_embeds = key_embeds

        # Compute attention and optimize edge weights
        output, attention_probs = self.attention(query_embed, key_embeds, value_embeds)
        return output, attention_probs

# Initialize model
model = QuantumGraphOptimizer(len(node_list), embedding_dim)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 300
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()

    # Select a random node as query
    query_node = torch.tensor([node_to_idx["cat"]])
    key_nodes = torch.tensor([node_to_idx[node] for node in node_list])

    # Target: The next node in the graph hierarchy
    target = torch.tensor([node_to_idx["animal"]])

    # Forward pass
    output, attention_probs = model(query_node, key_nodes)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Visualize loss curve
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Update graph with optimized edge weights
for i, (u, v) in enumerate(data):
    if i < attention_probs.size(1):  # Match size of edges and attention scores
        G[u][v]['weight'] = attention_probs[0, i].item()

import matplotlib.collections as mcoll

# Visualize optimized graph with red edges
pos = nx.spring_layout(G)  # Compute graph layout

plt.figure(figsize=(8, 6))

# Draw nodes and labels
nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='lightblue', edgecolors='black')
nx.draw_networkx_labels(G, pos, font_size=10)

# Draw edges in red
nx.draw_networkx_edges(G, pos, edge_color='red', width=2)  # Set edge color to red and increase width

plt.title("Optimized Graph with Red Edges")
plt.show()

# Evaluate clustering
node_embeddings = model.embedding(torch.tensor([node_to_idx[node] for node in node_list])).detach().numpy()
labels = [0 if 'animal' in node or 'living' in node else 1 for node in node_list]  # Sample labels
silhouette = silhouette_score(node_embeddings, labels)
print(f"Silhouette Score: {silhouette:.4f}")
