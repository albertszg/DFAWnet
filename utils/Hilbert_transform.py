import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Sequence, Union
from torch.autograd import Function
import torch.fft as fft
class HilbertTransform(nn.Module):
    """
    Determine the analytical signal of a Tensor along a particular axis.
    Requires PyTorch 1.7.0+ and the PyTorch FFT module (which is not included in NVIDIA PyTorch Release 20.10).
    Args:
        axis: Axis along which to apply Hilbert transform. Default 2 (first spatial dimension).
        N: Number of Fourier components (i.e. FFT size). Default: ``x.shape[axis]``.
    """

    def __init__(self, axis: int = 2, n: Union[int, None] = None) -> None:#Union[] 输入数据的类型 =输出数据的类型

        # if PT_BEFORE_1_7:
        #     raise InvalidPyTorchVersionError("1.7.0", self.__class__.__name__)

        super().__init__()
        self.axis = axis
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor or array-like to transform. Must be real and in shape ``[Batch, chns, spatial1, spatial2, ...]``.
        Returns:
            torch.Tensor: Analytical signal of ``x``, transformed along axis specified in ``self.axis`` using
            FFT of size ``self.N``. The absolute value of ``x_ht`` relates to the envelope of ``x`` along axis ``self.axis``.
        """

        # Make input a real tensor
        x = torch.as_tensor(x, device=x.device if isinstance(x, torch.Tensor) else None)
        if torch.is_complex(x):
            raise ValueError("x must be real.")
        x = x.to(dtype=torch.float)

        if (self.axis < 0) or (self.axis > len(x.shape) - 1):
            raise ValueError("Invalid axis for shape of x.")

        n = x.shape[self.axis] if self.n is None else self.n
        if n <= 0:
            raise ValueError("N must be positive.")
        x = torch.as_tensor(x, dtype=torch.complex64)
        # Create frequency axis
        f = torch.cat(
            [
                torch.true_divide(torch.arange(0, (n - 1) // 2 + 1, device=x.device), float(n)),
                torch.true_divide(torch.arange(-(n // 2), 0, device=x.device), float(n)),
            ]
        )
        xf = fft.fft(x, n=n, dim=self.axis)
        # Create step function
        u = torch.heaviside(f, torch.tensor([0.5], device=f.device))
        u = torch.as_tensor(u, dtype=x.dtype, device=u.device)
        new_dims_before = self.axis
        new_dims_after = len(xf.shape) - self.axis - 1
        for _ in range(new_dims_before):
            u.unsqueeze_(0)
        for _ in range(new_dims_after):
            u.unsqueeze_(-1)

        ht = fft.ifft(xf * 2 * u, dim=self.axis)

        # Apply transform
        return torch.as_tensor(ht, device=ht.device, dtype=ht.dtype)