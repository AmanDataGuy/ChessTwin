"""Stage 6: ResNet policy network.

6 residual blocks, 64 channels by default -- matches Maia Chess's proven size
(see plan_ahead.md Section 0/1 for why this size was chosen: it fits comfortably
on a 6GB GPU and there's no evidence a larger net is needed to prove this
project's hypothesis). Takes the Stage 3 board tensor (BOARD_PLANES, 8, 8) and
outputs raw logits over the move vocabulary -- legal-move masking is applied by
the caller via MoveVocabulary.legal_move_mask(), not inside the network.
"""

from __future__ import annotations

import torch
from torch import nn

from models.encoding import BOARD_PLANES

POLICY_HEAD_CHANNELS = 32


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return torch.relu(out)


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        channels: int = 64,
        num_blocks: int = 6,
        in_planes: int = BOARD_PLANES,
    ):
        super().__init__()
        self.stem_conv = nn.Conv2d(in_planes, channels, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_blocks)])

        self.policy_conv = nn.Conv2d(channels, POLICY_HEAD_CHANNELS, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(POLICY_HEAD_CHANNELS)
        self.policy_fc = nn.Linear(POLICY_HEAD_CHANNELS * 8 * 8, vocab_size)

    @classmethod
    def from_config(cls, config: dict, vocab_size: int) -> PolicyNetwork:
        model_config = config["model"]
        return cls(
            vocab_size=vocab_size,
            channels=model_config["channels"],
            num_blocks=model_config["residual_blocks"],
            in_planes=model_config["board_planes"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.stem_bn(self.stem_conv(x)))
        for block in self.blocks:
            out = block(out)

        policy = torch.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.flatten(1)
        return self.policy_fc(policy)
