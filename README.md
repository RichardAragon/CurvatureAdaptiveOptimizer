# CurvatureAdaptiveOptimizer

## Overview

`CurvatureAdaptiveOptimizer` is a PyTorch optimizer designed for machine learning models operating in dynamic geometric spaces, particularly hyperbolic and non-Euclidean manifolds. 

### Key Features
- Adaptive learning rate based on curvature
- Similar to Adam optimization strategy
- Flexible for hyperbolic and dynamic geometric models
- Handles weight decay
- Easy to integrate with PyTorch models

## Installation

```bash
pip install torch
# Copy the CurvatureAdaptiveOptimizer.py into your project
```

## Usage Example

```python
import torch
from curvature_adaptive_optimizer import CurvatureAdaptiveOptimizer

# Create your model
model = YourHyperbolicModel()

# Initialize optimizer
optimizer = CurvatureAdaptiveOptimizer(
    model.parameters(), 
    lr=0.001, 
    weight_decay=1e-5
)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = criterion(model(input), target)
    
    # Optional: pass current curvature
    optimizer.step(curvature=model.current_curvature)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License
