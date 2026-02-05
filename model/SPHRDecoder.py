import torch
from torch import nn, Tensor

from .DRAM import DetailRefinementAttentionModule


class DownSampler(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, scale: int):
		"""
		This class use convolutional layer to downsample the input image.
		:param in_channels: the input channels.
		:param out_channels: the output channels.
		:param scale: dawnsampling scale factor.
		"""
		super(DownSampler, self).__init__()
		self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=scale, padding=1)

	def forward(self, inputs: Tensor) -> Tensor:
		"""
		Downsample the input image.
		:param inputs: the input tensor, whose shape is (B, C, λH, λW), where λ denotes the
			downsampling factor.
		:return: the downsampled tensor, whose shape is (B, C, H, W).
		"""
		output = self.downsample(inputs)
		return output


class UpSampler(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, scale: int):
		"""
		This class use channel-wise upsampling and pixel shuffle to realize spacial upsampling.
		:param in_channels: input channels.
		:param out_channels: output channels.
		:param scale: upsampling scale.
		"""
		super(UpSampler, self).__init__()
		mid_channels = in_channels * (scale ** 2)
		self.conv_first = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
		self.pixel_shuffle = nn.PixelShuffle(scale)
		self.conv_last = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
	
	def forward(self, inputs: Tensor) -> Tensor:
		"""
		Doing forward propagation.
		:param inputs: the input tensor, whose shape is (B, C_in, H, W).
		:return: the output tensor, whose shape is (B, C_out, λH, λW).
		"""
		output = self.conv_first(inputs)
		output = self.pixel_shuffle(output)
		output = self.conv_last(output)
		return output


class SpatialRestorer(nn.Module):
	def __init__(self, in_channels: int, scale: int):
		"""
		Initialize the spatial restoring module.
		:param in_channels: the input channels.
		:param scale: the upsampling scale.
		"""
		super(SpatialRestorer, self).__init__()
		self.upsampler = UpSampler(in_channels, in_channels, scale)
	
	def forward(self, inputs: Tensor) -> Tensor:
		"""
		Get the spatial restored tensors.
		:param inputs: the input tensor, whose shape is (B, C, H, W).
		:return: the output tensor, whose shape is (B, C, λH, λW).
		"""
		output = self.upsampler(inputs)  # B, C, H, W
		return output


class DetailRestorer(nn.Module):
	def __init__(self, in_channels: int, num_features: int = 32, gamma: int = 4):
		"""
		This class restores the detail information.
		:param in_channels: the input channels.
		:param num_features: the number of features.
		:param gamma: the spectral enhancement factor within SEFFM.
		"""
		super(DetailRestorer, self).__init__()
		self.conv_feature = nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1)
		self.dram_down_1 = DetailRefinementAttentionModule(in_channels=num_features, gamma=gamma)
		self.dram_down_2 = DetailRefinementAttentionModule(in_channels=num_features * 2, gamma=gamma)
		self.dram_mid = DetailRefinementAttentionModule(in_channels=num_features * 4, gamma=gamma)
		self.dram_up_2 = DetailRefinementAttentionModule(in_channels=num_features * 2, gamma=gamma)
		self.dram_up_1 = DetailRefinementAttentionModule(in_channels=num_features * 2, gamma=gamma)
		self.dram_last = DetailRefinementAttentionModule(in_channels=num_features * 2, gamma=gamma)
		self.fully_connected = nn.Conv2d(
			num_features * 4, num_features * 2, kernel_size=1, stride=1, padding=0
		)
		self.downsample_1 = nn.Conv2d(
			num_features, num_features * 2, kernel_size=3, stride=2, padding=1
		)
		self.downsample_2 = nn.Conv2d(
			num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1
		)
		self.upsample_2 = UpSampler(
			in_channels=num_features * 4, out_channels=num_features * 2, scale=2
		)
		self.upsample_1 = UpSampler(
			in_channels=num_features * 2, out_channels=num_features, scale=2
		)
		self.conv_last = nn.Conv2d(
			num_features * 2, in_channels, kernel_size=3, stride=1, padding=1
		)
	
	def forward(self, inputs: Tensor) -> Tensor:
		"""
		Doing forward propagation.
		:param inputs: the input tensor, whose shape is (B, C, H, W).
		:return: the output tensor, whose shape is (B, C, H, W).
		"""
		output = self.conv_feature(inputs)  # B C_f H W
		res_1 = self.dram_down_1(output)  # B C_f H W
		output = self.downsample_1(res_1)  # B, 2C_f, H/2, W/2
		res_2 = self.dram_down_2(output)  # B, 2C_f, H/2, W/2
		output = self.downsample_2(output)  # B, 4C_f, H/4, W/4
		
		output = self.dram_mid(output)  # B, 4C_f, H/4, W/4
		
		output = self.upsample_2(output)  # B, 2C_f, H/2, W/2
		output = torch.cat((res_2, output), dim=1)  # B, 4C_f, H/2, W/2
		output = self.fully_connected(output)  # B, 2C_f, H/2, W/2
		output = self.dram_up_2(output)  # B, 2C_f, H/2, W/2
		output = self.upsample_1(output)  # B C_f H W
		output = torch.cat((res_1, output), dim=1)  # B 2C_f H W
		output = self.dram_up_1(output)  # B 2C_f H W
		output = self.dram_last(output)  # B 2C_f H W
		output = self.conv_last(output) + inputs  # B C H W
		return output
	

class SpectralRestorer(nn.Module):
	def __init__(self, in_channels: int, out_channels: int):
		"""
		Initialize the spectral restoring module.
		:param in_channels: the input channels.
		:param out_channels: the output channels.
		"""
		super(SpectralRestorer, self).__init__()
		self.upsampler = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
		self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
	
	def forward(self, inputs1: Tensor, inputs2: Tensor) -> Tensor:
		"""
		Get the spectral restored tensors.
		:param inputs1: the output of detail restorer, whose shape is (B, C_in, H, W).
		:param inputs2: the output of spatial restorer, whose shape is (B, C_in, H, W).
		:return: the output tensor, the shape is (B, C_out, H, W).
		"""
		output = self.upsampler(inputs1) + self.skip_connection(inputs2)  # B, C_out, H, W
		return output


class SPHRDecoder(nn.Module):
	def __init__(
			self, in_channels: int, out_channels: int, num_features: int = 32,
			scale: int = 4, gamma: int = 4,
	):
		"""
		The implementation of spatial-priority hierarchical reconstruction decoder.
		:param in_channels: the input channels.
		:param out_channels: the output channels.
		:param num_features: the number of features.
		:param scale: scale factor.
		:param gamma: the spectral enhancement factor within SEFFM.
		"""
		super(SPHRDecoder, self).__init__()
		self.spatial_restorer = SpatialRestorer(in_channels, scale)
		self.detail_restorer = DetailRestorer(in_channels, num_features, gamma)
		self.spectral_restorer = SpectralRestorer(in_channels, out_channels)
	
	def forward(self, inputs: Tensor) -> Tensor:
		"""
		Doing reconstruction for compressed HSIs.
		:param inputs: the compressed HSIs. whose shape is (B, C_in, H/4, W/4).
		:return: the reconstruction of the compressed HSIs, whose shape is (B, C_out, H, W).
		"""
		output1 = self.spatial_restorer(inputs)  # B, C_in, H, W
		output2 = self.detail_restorer(output1)  # B, C_in, H, W
		output = self.spectral_restorer(output2, output1)  # B, C_out, H, W
		return output
	

if __name__ == '__main__':
	model = SPHRDecoder(in_channels=27, out_channels=172)
	print(model)
	params = sum(p.numel() for p in model.parameters())
	print(f"Total number of parameters: {params / 1e6: .2f} M.")
	x = torch.randn(1, 27, 32, 1)
	print(f"The shape of input tensor is: {x.shape}.")
	y = model(x)
	print(f"The shape of output tensor is: {y.shape}.")
	