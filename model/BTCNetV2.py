import torch
from torch import nn, Tensor

from model.QuantizedConv2d import QuantConv2d
from model.SPHRDecoder import SPHRDecoder


class BTCNetV2(nn.Module):
	def __init__(
			self,
			original_channels: int,
			compressed_channels: int,
			quant_bit: int = 32,
			scale: int = 4,
			num_features: int = 32,
			gamma: int = 4,
	):
		"""
		This class is the final implementation of BTC-Net V2.
		:param original_channels: the original HSI's channels.
		:param compressed_channels: the compressed HSI's channels.
		:param scale: the scale factor.
		:param num_features: the number of features.
		:param gamma: the spectral enhancement factor within SEFFM.
		"""
		super(BTCNetV2, self).__init__()
		self.encoder = QuantConv2d(
			original_channels, compressed_channels, kernel_size=(11, 4), stride=4, padding=(5, 0),
			quant_bit=quant_bit,
		)
		self.decoder = SPHRDecoder(
			in_channels=compressed_channels, out_channels=original_channels, num_features=num_features,
			scale=scale, gamma=gamma
		)
	
	def forward(self, inputs: Tensor) -> Tensor:
		"""
		Doing compression and reconstruction process.
		:param inputs: the original HSI, whose shape is (B, C, H, W).
		:return: the reconstructed HSI, whose shape is (B, C, H, W).
		"""
		output = self.decoder(self.encoder(x)[0])
		return output


if __name__ == '__main__':
	model = BTCNetV2(172, 27, 8, )
	print(model)
	params = sum(p.numel() for p in model.parameters())
	print(f"Total number of parameters: {params / 1e6: .2f} M.")
	x = torch.randn(1, 172, 128, 4)
	print(f"The shape of input tensor is: {x.shape}.")
	y = model(x)
	print(f"The shape of output tensor is: {y.shape}.")
	

