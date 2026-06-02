import time
import struct
from typing import Union, List, Tuple, Optional

import torch
import torch.nn as nn


class Node:
    """Huffman tree node"""

    def __init__(self, symbol: int, prob: float, left: 'Node' = None, right: 'Node' = None, code: str = None):
        self._symbol = symbol
        self._prob = prob
        self._left = left
        self._right = right
        self._code = code

    def __repr__(self) -> str:
        return (f"Node(symbol={self.symbol}, prob={self.prob:.6f}, "
                f"left={self.left is not None}, right={self.right is not None}, code={self.code})")

    @property
    def symbol(self) -> int:
        return self._symbol

    @property
    def prob(self) -> float:
        return self._prob

    @property
    def left(self) -> 'Node':
        return self._left

    @property
    def right(self) -> 'Node':
        return self._right

    @property
    def code(self) -> str:
        return self._code

    @symbol.setter
    def symbol(self, value: int) -> None:
        if value >= 0:
            self._symbol = value
        else:
            raise ValueError(f'Symbol must be non-negative integer, got {value}.')

    @prob.setter
    def prob(self, value: float) -> None:
        if 0.0 <= value <= 1.0:
            self._prob = value
        else:
            raise ValueError(f'Probability must be in [0, 1], got {value}.')

    @left.setter
    def left(self, node: 'Node') -> None:
        if isinstance(node, Node):
            self._left = node
        else:
            raise TypeError(f'Left child must be a Node instance, got {type(node)}.')

    @right.setter
    def right(self, node: 'Node') -> None:
        if isinstance(node, Node):
            self._right = node
        else:
            raise TypeError(f'Right child must be a Node instance, got {type(node)}.')

    @code.setter
    def code(self, value: str) -> None:
        if all(ch in '01' for ch in value):
            self._code = value
        else:
            raise ValueError(f'Code must be a binary string, got {value}.')


class HuffmanTree:
    """Build Huffman tree from a probability list"""

    def __init__(self, probs: List[float]):
        self.probs = probs
        self.root: Node = self._create_tree()
        self._assign_codes(self.root, "")

    def __len__(self) -> int:
        return len(self.probs)

    def __call__(self) -> Node:
        return self.root

    def _create_tree(self) -> Node:
        nodes = [Node(idx, prob) for idx, prob in enumerate(self.probs)]
        while len(nodes) > 1:
            nodes.sort(key=lambda node: (node.prob, node.symbol))
            left, right = nodes[0], nodes[1]
            parent = Node(symbol=-1, prob=left.prob + right.prob, left=left, right=right)
            nodes.pop(0)
            nodes.pop(0)
            nodes.append(parent)
        return nodes[0]

    def _assign_codes(self, node: Node, code: str) -> None:
        node.code = code
        if node.left and node.right:
            self._assign_codes(node.left, code + "0")
            self._assign_codes(node.right, code + "1")


class HuffmanCodex:
    """Huffman encoder/decoder (using fixed probability table)"""

    def __init__(self):
        self.tree: Optional[HuffmanTree] = None
        self.leaf_nodes: List[Node] = []

    def _collect_leaves(self, node: Node) -> None:
        if node.left and node.right:
            self._collect_leaves(node.left)
            self._collect_leaves(node.right)
        else:
            self.leaf_nodes.append(node)

    def encode(self, symbol: Union[List[int], int], probs: List[float], bit_stream: str) -> str:
        self.tree = HuffmanTree(probs)
        self.leaf_nodes = []
        self._collect_leaves(self.tree.root)

        if isinstance(symbol, int):
            for leaf in self.leaf_nodes:
                if leaf.symbol == symbol:
                    return bit_stream + leaf.code
            raise ValueError(f"Symbol {symbol} not found in Huffman tree.")
        else:
            self.leaf_nodes.sort(key=lambda node: node.symbol)
            for sym in symbol:
                bit_stream += self.leaf_nodes[sym].code
            return bit_stream

    def decode(self, probs: List[float], bit_stream: str, length: int) -> Tuple[Union[List[int], int], str]:
        self.tree = HuffmanTree(probs)
        self.leaf_nodes = []
        self._collect_leaves(self.tree.root)

        decoded = []
        remaining = bit_stream
        for _ in range(length):
            matched = False
            for leaf in self.leaf_nodes:
                if remaining.startswith(leaf.code):
                    decoded.append(leaf.symbol)
                    remaining = remaining[len(leaf.code):]
                    matched = True
                    break
            if not matched:
                raise RuntimeError(f"Failed to decode from bit stream: {remaining[:20]}...")

        if length == 1:
            return decoded[0], remaining
        return decoded, remaining


class EntropyCodex:
    """Entropy codec (without hyperprior, only Huffman coding based on frequency table)"""

    def __init__(
        self,
        quantize_bit: int = 8,
        device: str = "cpu",
        model: nn.Module = None
    ):
        self.huffman_codex = HuffmanCodex()
        self.quantize_bit = quantize_bit
        self.device = device
        self.model = model
        if model is not None:
            self.model.eval()

    def set_model(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    @staticmethod
    def get_freq_table(int_tensor: torch.Tensor) -> List[float]:
        """Compute frequency table (normalized probabilities) from an integer tensor"""
        counts = torch.bincount(int_tensor.flatten())
        probs = counts.to(torch.float32) / int_tensor.numel()
        # Use a small lower bound to avoid zero probabilities
        return torch.clamp(probs, min=1e-8, max=1.0).tolist()

    @staticmethod
    def read_bits(bit_stream: str, length: int) -> Tuple[str, str]:
        return bit_stream[:length], bit_stream[length:]

    @staticmethod
    def int8_to_binary(value: int) -> str:
        if not (0 <= value <= 255):
            raise ValueError(f"Value out of range for int8: {value}")
        return format(value, '08b')

    @staticmethod
    def binary_to_int8(binary_str: str) -> int:
        if not set(binary_str).issubset('01'):
            raise ValueError(f"Invalid binary string: {binary_str}")
        return int(binary_str, 2)

    @staticmethod
    def float32_to_binary(value: float) -> str:
        packed = struct.pack('!f', value)
        return ''.join(f'{byte:08b}' for byte in packed)

    @staticmethod
    def binary_to_float32(binary: str) -> float:
        if len(binary) != 32:
            raise ValueError("Binary string must be 32 bits long.")
        int_val = int(binary, 2)
        packed = int_val.to_bytes(4, byteorder='big')
        return struct.unpack('!f', packed)[0]

    @staticmethod
    def to_symbol(
        tensor_: torch.Tensor,
        min_: Union[torch.Tensor, float],
        quantize_step: Union[torch.Tensor, float]
    ) -> torch.Tensor:
        int_tensor = (tensor_ - min_) / quantize_step
        return int_tensor.round().to(torch.int)

    @staticmethod
    def quantize(
        int_tensor: torch.Tensor,
        min_: Union[torch.Tensor, float],
        quantize_step: Union[torch.Tensor, float]
    ) -> torch.Tensor:
        return int_tensor * quantize_step + min_

    @staticmethod
    def compute_bpp(binary_length: int, x_shape: torch.Size) -> float:
        batch, channels, height, width = x_shape
        return binary_length / (batch * channels * height * width)

    def cat_full_image(self, tensor_: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = tensor_.shape
        reshaped = torch.zeros((height, width * batch, channels), device=self.device, dtype=torch.float32)
        for i in range(batch):
            reshaped[:, i * width:(i + 1) * width, :] = tensor_[i].permute(1, 2, 0)
        return reshaped

    def compute_psnr(self, original: torch.Tensor, restored: torch.Tensor) -> float:
        orig = self.cat_full_image(original).reshape(-1, original.shape[1])
        rest = self.cat_full_image(restored).reshape(-1, restored.shape[1])
        mse = torch.mean((orig - rest) ** 2, dim=0, keepdim=True)
        max_val, _ = torch.max(orig, dim=0, keepdim=True)
        return torch.mean(10 * torch.log10((max_val ** 2) / mse)).item()

    def compute_sam(self, original: torch.Tensor, restored: torch.Tensor) -> float:
        orig = self.cat_full_image(original)
        rest = self.cat_full_image(restored)
        num = torch.sum(orig * rest, dim=2)
        den = torch.sqrt(torch.sum(rest ** 2, dim=2) * torch.sum(orig ** 2, dim=2))
        cos_theta = torch.clamp(num / den, min=-1.0, max=1.0)
        angle_deg = torch.rad2deg(torch.acos(cos_theta))
        return (torch.sum(angle_deg) / (orig.shape[0] * orig.shape[1])).item()

    def compute_rmse(self, original: torch.Tensor, restored: torch.Tensor) -> float:
        orig = self.cat_full_image(original)
        rest = self.cat_full_image(restored)
        mse = torch.mean((rest - orig) ** 2, dim=(0, 1))
        return torch.sqrt(torch.mean(mse)).item()

    def compress_list(self, int_tensor: torch.Tensor, freq_table: List[float], bit_stream: str) -> str:
        """Huffman encode an integer tensor using a precomputed frequency table"""
        symbols = int_tensor.flatten().tolist()
        return self.huffman_codex.encode(symbols, freq_table, bit_stream)

    def decompress_list(
        self,
        freq_table: List[float],
        bit_stream: str,
        min_: torch.Tensor,
        quantize_step: torch.Tensor,
        shape: Tuple[int, int, int, int]
    ) -> Tuple[torch.Tensor, str]:
        """Decode and dequantize"""
        b, c, h, w = shape
        symbols, remaining = self.huffman_codex.decode(freq_table, bit_stream, b * c * h * w)
        int_tensor = torch.tensor(symbols, device=self.device, dtype=torch.float32).reshape(shape)
        return int_tensor * quantize_step + min_, remaining

    def run(self, x: torch.Tensor, norm_max: float, norm_min: float) -> Tuple[torch.Tensor, float, float, float, float]:
        """
        Full compression and decompression pipeline without hyperprior
        """
        # If quantization bit width is 32, directly use floating point representation (no entropy coding)
        if self.quantize_bit == 32:
            start = time.time()
            y = self.model.encoder(x)
            if isinstance(y, tuple):
                y = y[0]
            print(f"Encoder time: {time.time() - start:.4f}s")
            restored = self.model.decoder(y)
            restored = restored * (norm_max - norm_min) + norm_min
            x = x * (norm_max - norm_min) + norm_min
            bpp = y.numel() * 32 / x.numel()
            psnr = self.compute_psnr(x, restored)
            sam = self.compute_sam(x, restored)
            rmse = self.compute_rmse(x, restored)
            return restored, bpp, psnr, sam, rmse

        x = x.contiguous()
        start = time.time()

        # Encode to obtain latent representation
        y = self.model(x, return_encoded=True)[1]  # Assuming model returns (output, encoded)
        if isinstance(y, tuple):
            # Some models may return additional information
            y, int_y, max_, min_ = y
            int_y = int_y.to(torch.int)
        else:
            max_ = torch.max(y)
            min_ = torch.min(y)
            quantize_step = (max_ - min_) / (2 ** self.quantize_bit - 1)
            int_y = self.to_symbol(y, min_, quantize_step)

        b, c, h, w = y.shape

        # Build bitstream header: shape, max, min
        bit_stream = (
            self.int8_to_binary(b) +
            self.int8_to_binary(c) +
            self.int8_to_binary(h) +
            self.int8_to_binary(w) +
            self.float32_to_binary(max_.item()) +
            self.float32_to_binary(min_.item())
        )

        # Compute and write frequency table
        freq_table = self.get_freq_table(int_y)
        for prob in freq_table:
            bit_stream += self.float32_to_binary(prob)

        # Compress integer tensor
        bit_stream = self.compress_list(int_y, freq_table, bit_stream)
        print(f"Compression time: {time.time() - start:.4f}s")

        total_bits = len(bit_stream)

        # Decode header information
        bin_b, bit_stream = self.read_bits(bit_stream, 8)
        bin_c, bit_stream = self.read_bits(bit_stream, 8)
        bin_h, bit_stream = self.read_bits(bit_stream, 8)
        bin_w, bit_stream = self.read_bits(bit_stream, 8)
        bin_max, bit_stream = self.read_bits(bit_stream, 32)
        bin_min, bit_stream = self.read_bits(bit_stream, 32)

        b_rec = self.binary_to_int8(bin_b)
        c_rec = self.binary_to_int8(bin_c)
        h_rec = self.binary_to_int8(bin_h)
        w_rec = self.binary_to_int8(bin_w)
        max_rec = self.binary_to_float32(bin_max)
        min_rec = self.binary_to_float32(bin_min)
        max_t = torch.tensor(max_rec, device=self.device, dtype=torch.float32)
        min_t = torch.tensor(min_rec, device=self.device, dtype=torch.float32)
        q_step = (max_t - min_t) / (2 ** self.quantize_bit - 1)

        # Read frequency table
        freq_table_rec = []
        for _ in range(2 ** self.quantize_bit):
            bin_prob, bit_stream = self.read_bits(bit_stream, 32)
            freq_table_rec.append(self.binary_to_float32(bin_prob))

        # Decompress
        decoded_int, bit_stream = self.decompress_list(
            freq_table_rec, bit_stream, min_t, q_step, (b_rec, c_rec, h_rec, w_rec)
        )
        assert len(bit_stream) == 0, f"Remaining bits after decompression: {len(bit_stream)}"

        # Decode to recover image
        restored = self.model.decoder(decoded_int)
        restored = restored * (norm_max - norm_min) + norm_min
        x = x * (norm_max - norm_min) + norm_min

        bpp = self.compute_bpp(total_bits, x.shape)
        psnr = self.compute_psnr(x, restored)
        sam = self.compute_sam(x, restored)
        rmse = self.compute_rmse(x, restored)

        return restored, bpp, psnr, sam, rmse
