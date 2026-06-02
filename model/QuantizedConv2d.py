import torch
from torch import nn, Tensor
from torch.nn import functional as F


class QuantConv2d(nn.Conv2d):
	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			kernel_size: int | tuple[int, int],
			stride: int | tuple[int, int],
			padding: int | tuple[int, int],
			quant_bit: int = 32,
			lower_bound: float = 1e-10,
	):
		"""
		Initializes a QuantConv2d layer.
		:param in_channels: input channels.
		:param out_channels: output channels.
		:param kernel_size: kernel size.
		:param stride: stride.
		:param padding: padding.
		:param quant_bit: quantization bits.
		:param lower_bound: lower bound to avoid very small values.
		"""
		super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding)
		self.quant_bit = quant_bit
		self.lower_bound = lower_bound
		
		self.p_relu = nn.PReLU()

	def uniform_quantize(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
		"""
		Uniformly quantize the input tensor.
		:param inputs: tensor to be quantized.
		:return: a tuple of 2 elements:
			1: the quantized tensor.
			2: the integer form of input tensor.
		"""
		# if tensor contains only one value, no need to quantize
		if inputs.numel() == 1:
			return inputs, torch.zeros_like(inputs, device=inputs.device, dtype=torch.int)
		
		max_, min_ = torch.max(inputs), torch.min(inputs)
		quant_step = (max_ - min_) / (2 ** self.quant_bit - 1)
		
		# quantize the input tensor to integer values using straight-through-estimation (STE)
		int_tensor = (inputs - min_) / quant_step
		int_tensor = (int_tensor.round() - int_tensor).detach() + int_tensor
		
		# dequantization
		output = int_tensor * quant_step + min_
		return output, int_tensor.to(torch.int)

	def forward(self, inputs: Tensor, return_5d: bool = False) -> tuple[Tensor, Tensor | None]:
		"""
		Doing convolution.
		:param inputs: tensor of size 5: (B, N, C, H, W).
		:return: a tuple of 5 elements:
			1: the quantized output, shape is: (B * N, C, H, W).
			2: the integer form of the output.
		"""
		b, c, h, w = inputs.shape
		
		# if quant_bit is set to 32, no more quantization will be operated
		if self.quant_bit == 32:
			output = self.p_relu(super().forward(inputs))
			return output, None
		
		# uniformly quantize the weight and bias of this layer
		weight, _ = self.uniform_quantize(self.weight)
		bias, _ = self.uniform_quantize(self.bias)
		
		# uniformly quantize the input tensor
		inputs, _ = self.uniform_quantize(inputs)
		
		# because we use quantized convolution here, super().forward() is not available
		output = F.conv2d(inputs, weight, bias, self.stride, self.padding, self.dilation, self.groups)
		output = self.p_relu(output)
		
		# uniformly quantize the output tensor
		output, int_tensor = self.uniform_quantize(output)
		return output, int_tensor


if __name__ == '__main__':
	model = QuantConv2d(
		172, 27, (11, 4), (4, 1), (5, 0), 8
	)
	print(model)
	params = sum(p.numel() for p in model.parameters())
	print(f"Total number of parameters: {params / 1e6: .2f} M.")
	x = torch.randn(2, 172, 256, 4)
	print(f"The shape of input tensor is: {x.shape}.")
	quant_y, int_y = model(x)
	print(f"The shape of output tensor is: {quant_y.shape}, the type is {quant_y.dtype}.")
	print(f"The shape of integer-form output tensor is: {int_y.shape}, the type is {int_y.dtype}.")
	