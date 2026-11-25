import torch


class _STEBinary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x.sign()
        # torch.where instead of y[y == 0] = 1: same result, but the masked assignment forces a device sync on MPS
        y = torch.where(y == 0, torch.ones_like(y), y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        deriv = ((x > -1) & (x < 1))
        grad_output = grad_output * deriv
        return grad_output


class _SmoothSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=100):
        ctx.alpha = alpha
        ctx.save_for_backward(x)
        y = x.sign()
        # torch.where instead of y[y == 0] = 1: same result, but the masked assignment forces a device sync on MPS
        y = torch.where(y == 0, torch.ones_like(y), y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output * alpha * (1 - torch.tanh(alpha * x)**2)
        return grad_input, None


STEBinary = _STEBinary.apply
SmoothSign = _SmoothSign.apply
