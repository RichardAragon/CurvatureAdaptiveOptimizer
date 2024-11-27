import math
import torch


class CurvatureAdaptiveOptimizer(torch.optim.Optimizer):
    """
    Curvature-Adaptive Optimizer for Hyperbolic and Dynamic Geometric Spaces

    Adapts learning rate based on curvature, with features similar to Adam optimizer.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): Learning rate. Defaults to 1e-3.
        betas (Tuple[float, float], optional): Coefficients used for computing running averages of gradient 
            and its square. Defaults to (0.9, 0.999).
        eps (float, optional): Term added to the denominator to improve numerical stability. Defaults to 1e-8.
        weight_decay (float, optional): Weight decay (L2 penalty). Defaults to 0.

    Example:
        >>> optimizer = CurvatureAdaptiveOptimizer(model.parameters(), lr=0.001)
        >>> for epoch in range(num_epochs):
        ...     optimizer.zero_grad()
        ...     loss = criterion(model(input), target)
        ...     optimizer.step(curvature=current_curvature)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, curvature=None, closure=None):
        """
        Performs a single optimization step.

        Args:
            curvature (float, optional): Current curvature for adaptive learning rate scaling.
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            float: Loss value if closure is provided.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Fetch gradient
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Unpack hyperparameters
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Decay running averages
                state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
                state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute denominator
                denom = (state['exp_avg_sq'].sqrt() / math.sqrt(1 - beta2 ** state['step'])) + group['eps']

                # Adaptive learning rate calculation
                if curvature is not None:
                    # Curvature-based adaptive scaling
                    curvature_factor = 1.0 / (1 + abs(curvature))
                    step_size = group['lr'] * curvature_factor * \
                        (math.sqrt(1 - beta2 ** state['step']) / (1 - beta1 ** state['step']))
                else:
                    step_size = group['lr'] * \
                        (math.sqrt(1 - beta2 ** state['step']) / (1 - beta1 ** state['step']))

                # Apply weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Update parameters
                p.data.addcdiv_(state['exp_avg'], denom, value=-step_size)

        return loss
