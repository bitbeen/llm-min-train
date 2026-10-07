import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """把图片切成小块，每块线性映射成一个和文字同维的向量。"""

    def __init__(self, image_size, patch_size, d_model, in_channels=3):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size 必须能被 patch_size 整除")
        self.n_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, images):
        x = self.proj(images)
        return x.flatten(2).transpose(1, 2)


def solid_image(color, size):
    """一张纯色图。红、绿、蓝分别写进三个通道里的一个，数值是 0 或 1。"""
    image = torch.zeros(3, size, size)
    channel = {"红": 0, "绿": 1, "蓝": 2}[color]
    image[channel] = 1.0
    return image


def split_image(left, right, size):
    """左半和右半各一种颜色，用来学位置，而不只是整张纯色。"""
    image = torch.zeros(3, size, size)
    half = size // 2
    image[{"红": 0, "绿": 1, "蓝": 2}[left], :, :half] = 1.0
    image[{"红": 0, "绿": 1, "蓝": 2}[right], :, half:] = 1.0
    return image


def make_image(row, size):
    """layout 为 split 时画左右两色，否则画纯色。"""
    if row.get("layout") == "split":
        return split_image(row["left"], row["right"], size)
    return solid_image(row["color"], size)
