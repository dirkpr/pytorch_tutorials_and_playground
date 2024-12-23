#!/bin/env python3
"""Follow building autograd tutorial from https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html#."""

import torch


def simple_autograd() -> None:
    """Demonstrate automatic differentiation with PyTorch.

    Use simple linear regression model to demonstrate autograd as used in regression problems."""
    x = 0.5 * torch.ones(5)  # input tensor
    x.requires_grad_()
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    z = torch.matmul(x, w) + b

    loss = z.sum()
    loss.backward()

    print("w = ", w)

    # We expect this to be equal to x
    print("Gradient w.r.t. w", w.grad)

    # Check if all close to x. Need to cast x into correct shape
    x_reshaped = x.unsqueeze(1).expand_as(w)  # shape: (5,3)
    print("All close to x:", torch.allclose(w.grad, x_reshaped))

    # We expect this to be equal to 1
    print("Gradient w.r.t. b", b.grad)
    print("All close to 1:", torch.allclose(b.grad, torch.ones(3)))

    print("Gradient w.r.t. x", x.grad)
    print("Gradient is equal to sum of w elements along dim 1:", torch.allclose(x.grad, w.sum(1)))

    print(f"Gradient function for z = {z.grad_fn}")
    print(f"Gradient function for loss = {loss.grad_fn}")


def bce_autograd() -> None:
    """Demonstrate automatic differentiation with PyTorch.

    Use binary cross entropy loss function to demonstrate autograd as used in classification problems.
    """

    x = torch.ones(5)  # input tensor
    y = torch.zeros(3)  # expected output
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    z = torch.matmul(x, w) + b

    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

    loss.backward()
    print(w.grad)
    print(b.grad)
    print(f"Gradient function for z = {z.grad_fn}")
    print(f"Gradient function for loss = {loss.grad_fn}")

    # Disable gradient tracking
    z = torch.matmul(x, w) + b
    print(z.requires_grad)

    if False:
        with torch.no_grad():
            z = torch.matmul(x, w) + b
        print(z.requires_grad)
    else:
        z_det = z.detach()
        print(z_det.requires_grad)


def jabobian_product() -> None:
    """Calculate Jabobian product with pytorch."""
    print("\n\nJacobian product")
    inp = torch.eye(4, 5, requires_grad=True)
    out = (inp + 1).pow(2).t()
    print("before back propagation: out == \n", out, "\n")
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"First call\n{inp.grad}")
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"\nSecond call\n{inp.grad}")
    inp.grad.zero_()
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"\nCall after zeroing gradients\n{inp.grad}")


def main() -> None:
    """Demonstrate automatic differentiation with PyTorch."""
    simple_autograd()
    bce_autograd()
    jabobian_product()


if __name__ == "__main__":
    main()
