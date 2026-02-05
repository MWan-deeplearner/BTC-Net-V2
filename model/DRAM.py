from einops import rearrange
import torch
from torch import nn, Tensor
from torch.nn import functional as F


class SpatialEnhancementSelfAttentionModule(nn.Module):
	def __init__(self, in_channels: int):
		"""
		Initializes the SESAM module.
		:param in_channels: input channels.
		"""
		super(SpatialEnhancementSelfAttentionModule, self).__init__()
		self.in_channels = in_channels
		
		self.layer_norm = nn.LayerNorm(normalized_shape=in_channels)
		self.module_q = self.make_layers()
		self.module_k = self.make_layers()
		self.module_v = self.make_layers()
		self.alpha = nn.Parameter(torch.tensor(1.0))
		self.linear_last = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
	
	def make_layers(self) -> nn.Module:
		"""
		Create a single convolutional neural network.
		:return: the convolutional neural network.
		"""
		return nn.Sequential(
			nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1)
		)
	
	def forward(self, inputs: Tensor) -> Tensor:
		"""
		Doing forward propagation.
		:param inputs: the input tensor, whose shape is (B, C, H, W).
		:return: the output tensor, whose shape is (B, C, H, W).
		"""
		b, c, h, w = inputs.shape
		output = rearrange(inputs, "b c h w -> b h w c")  # B, H, W, C
		output = self.layer_norm(output)  # B, H, W, C
		output = rearrange(output, "b h w c -> b c h w")  # B, C, H, W
		
		q = self.module_q(output).permute(0, 2, 3, 1).reshape(b, -1, c)  # B (H W) C
		k = self.module_k(output).reshape(b, c, -1)  # B C (H W)
		v = self.module_v(output).permute(0, 2, 3, 1).reshape(b, -1, c)  # B (H W) C
		
		k_mat_q = F.softmax(torch.matmul(k, q) / self.alpha, dim=-1)  # B C C
		output = torch.matmul(v, k_mat_q).reshape(b, h, w, c).permute(0, 3, 1, 2)  # B, C, H, W
		output = self.linear_last(output) + inputs  # B, C, H, W
		return output


class SpectralEnhancementFeedForwardModule(nn.Module):
	def __init__(self, in_channels: int, gamma: int = 4):
		"""
		Initializes the SEFFM module.
		:param in_channels: input channels.
		:param gamma: spectral enhancement factor.
		"""
		super(SpectralEnhancementFeedForwardModule, self).__init__()
		self.in_channels = in_channels
		self.gamma = gamma
		self.mid_channels = self.in_channels * self.gamma
		
		self.layer_norm = nn.LayerNorm(normalized_shape=in_channels)
		self.layer_1 = self.make_layers()
		self.layer_2 = self.make_layers()
		self.linear_last = nn.Conv2d(self.mid_channels, in_channels, kernel_size=1, stride=1, padding=0)
	
	def make_layers(self):
		"""
		Create a single convolutional neural network.
		:return: the convolutional neural network.
		"""
		module = nn.Sequential(
			nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=1, stride=1, padding=0),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1)
		)
		return module
	
	def forward(self, inputs: Tensor) -> Tensor:
		"""
		Doing forward propagation.
		:param inputs: the input tensor, whose shape is (B, C, H, W).
		:return: the output tensor, whose shape is (B, C, H, W).
		"""
		output = rearrange(inputs, "b c h w -> b h w c")  # B, H, W, C
		output = self.layer_norm(output)  # B, H, W, C
		output = rearrange(output, "b h w c -> b c h w")  # B, C, H, W
		
		mid_1 = self.layer_1(output)  # B, C, H, W
		mid_2 = self.layer_2(output)  # B, C, H, W
		
		output = mid_1 * F.leaky_relu(mid_2, negative_slope=0.2)  # B, C, H, W
		output = self.linear_last(output) + inputs  # B, C, H, W
		return output


class DetailRefinementAttentionModule(nn.Module):
	def __init__(self, in_channels: int, gamma: int = 4):
		"""
		Initializes the DRAM module.
		:param in_channels: the input channels.
		:param gamma: spectral enhancement factor.
		"""
		super(DetailRefinementAttentionModule, self).__init__()
		self.SESAM = SpatialEnhancementSelfAttentionModule(in_channels=in_channels)
		self.SEFFM = SpectralEnhancementFeedForwardModule(in_channels=in_channels, gamma=gamma)

	def forward(self, inputs: Tensor) -> Tensor:
		"""
		Doing forward propagation.
		:param inputs: the input tensor, whose shape is (B, C, H, W).
		:return: the output tensor, whose shape is (B, C, H, W).
		"""
		output = self.SEFFM(self.SESAM(inputs))  # B, C, H, W
		return output


if __name__ == '__main__':
	model = DetailRefinementAttentionModule(in_channels=172)
	print(model)
	params = sum(p.numel() for p in model.parameters())
	print(f"Total number of parameters: {params / 1e6: .2f} M.")
	x = torch.randn(1, 172, 128, 4)
	print(f"The shape of input tensor is: {x.shape}.")
	y = model(x)
	print(f"The shape of output tensor is: {y.shape}.")
	
	