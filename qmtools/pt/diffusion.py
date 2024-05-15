
import torch
from torch import nn


def _get_padding(kernel_size: int | tuple[int, ...], nd: int) -> tuple[int, ...]:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * nd
    padding = []
    for i in range(nd):
        padding += [(kernel_size[i] - 1) // 2]
    return tuple(padding)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...] = 3,
        depth: int = 2,
        padding_mode: str = "zeros",
        res_connection: bool = True,
        activation: bool = None,
        last_activation: bool = True,
    ):
        super().__init__()

        self.act = activation or nn.ReLU()
        if last_activation:
            self.acts = [self.act] * depth
        else:
            self.acts = [self.act] * (depth - 1) + [nn.Identity()]

        padding = _get_padding(kernel_size, 3)
        self.convs = nn.ModuleList(
            [nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode)]
        )
        for _ in range(depth - 1):
            self.convs.append(
                nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode)
            )

        if res_connection:
            if in_channels != out_channels:
                self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
            else:
                self.res_conv = nn.Identity()
        else:
            self.res_conv = None

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = x_in
        for conv, act in zip(self.convs, self.acts):
            x = act(conv(x))
        if self.res_conv:
            x = x + self.res_conv(x_in)
        return x


class DensityDiffusionVAE(nn.Module):

    def __init__(self, device: str | torch.device = "cpu"):
        super().__init__()

        self.act = nn.ReLU()
        self.encoder = nn.Sequential(
            # input_shape = (n_batch, 1, nx, ny, nz)
            ResBlock(1, 8, kernel_size=3, depth=2, padding_mode="circular", activation=self.act),
            ResBlock(8, 8, kernel_size=3, depth=2, padding_mode="circular", activation=self.act),
            # (n_batch, 8, nx, ny, nz)
            nn.Conv3d(8, 8, kernel_size=2, stride=2, padding=0),
            # (n_batch, 8, nx / 2, ny / 2, nz / 2)
            ResBlock(8, 32, kernel_size=3, depth=2, padding_mode="circular", activation=self.act),
            ResBlock(32, 32, kernel_size=3, depth=2, padding_mode="circular", activation=self.act),
            # (n_batch, 32, nx / 2, ny / 2, nz / 2)
            nn.Conv3d(32, 32, kernel_size=2, stride=2, padding=0),
            # (n_batch, 32, nx / 4, ny / 4, nz / 4)
            ResBlock(32, 64, kernel_size=3, depth=2, padding_mode="circular", activation=self.act),
            ResBlock(64, 64, kernel_size=3, depth=2, padding_mode="circular", activation=self.act),
            # (n_batch, 64, nx / 4, ny / 4, nz / 4)
            nn.Conv3d(64, 8, kernel_size=3, padding=1, padding_mode="circular"),
        )
        self.decoder = nn.Sequential(
            # input_shape = (n_batch, 4, nx / 4, ny / 4, nz / 4)
            ResBlock(4, 64, kernel_size=3, depth=2, padding_mode="circular", activation=self.act),
            ResBlock(64, 64, kernel_size=3, depth=2, padding_mode="circular", activation=self.act),
            # (n_batch, 64, nx / 4, ny / 4, nz / 4)
            nn.Upsample(scale_factor=2, mode="trilinear"),
            # (n_batch, 64, nx / 2, ny / 2, nz / 2)
            ResBlock(64, 32, kernel_size=3, depth=2, padding_mode="circular", activation=self.act),
            ResBlock(32, 32, kernel_size=3, depth=2, padding_mode="circular", activation=self.act),
            # (n_batch, 32, nx / 2, ny / 2, nz / 2)
            nn.Upsample(scale_factor=2, mode="trilinear"),
            # (n_batch, 32, nx, ny, nz)
            ResBlock(32, 8, kernel_size=3, depth=2, padding_mode="circular", activation=self.act),
            ResBlock(8, 8, kernel_size=3, depth=2, padding_mode="circular", activation=self.act),
            # (n_batch, 8, nx, ny, nz)
            nn.Conv3d(8, 1, kernel_size=3, padding=1, padding_mode="circular"),
            # (n_batch, 1, nx, ny, nz)
        )

        self.device = device
        self.to(device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (n_batch, nx, ny, nz)
        x = x.unsqueeze(1)  # x.shape = (n_batch, 1, nx, ny, nz)
        x = self.encoder(x)  # x.shape = (n_batch, 8, nx / 4, ny / 4, nz / 4)
        mean, log_var = x.chunk(2, dim=1)  # mean.shape = log_var.shape = (n_batch, 4, nx / 4, ny / 4, nz / 4)
        return mean, log_var

    def decode(self, x: torch.Tensor, out_relu: bool = True) -> torch.Tensor:
        x = self.decoder(x)  # x.shape = (n_batch, 1, nx, ny, nz)
        x = x.squeeze(1)  # x.shape = (n_batch, nx, ny, nz)
        if out_relu:
            x = nn.functional.relu(x)
        return x

    def sample_latent_space(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        log_var = torch.clamp(log_var, -30, 20)
        std = log_var.exp().sqrt()
        x = mean + std * torch.normal(torch.zeros_like(mean), torch.ones_like(std))
        return x

    def forward(self, x, out_relu: bool = True):
        mean, log_var = self.encode(x)  # mean.shape = log_var.shape = (n_batch, 4, nx / 4, ny / 4, nz / 4)
        x = self.sample_latent_space(mean, log_var)  # x.shape = (n_batch, 4, nx / 4, ny / 4, nz / 4)
        x = self.decode(x, out_relu=out_relu)  # x.shape = (n_batch, nx, ny, nz)
        return x, mean, log_var


class VAELoss(nn.Module):

    def __init__(self, kl_weight=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.kl_weight = kl_weight

    def forward(self, pred, ref):
        pred_rec, mean, log_var = pred
        reconstruction_loss = self.mse(pred_rec, ref).mean()
        var = log_var.exp()
        kl_loss = 0.5 * (mean**2 + var - log_var - 1).sum(dim=1).mean()
        loss = reconstruction_loss + self.kl_weight * kl_loss
        return [loss, reconstruction_loss, kl_loss]


class UnetResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        t_embed_size: int,
        kernel_size: int | tuple[int, ...] = 3,
        depth: int = 2,
        padding_mode: str = "zeros",
        activation: bool = None,
    ):
        super().__init__()

        self.x_conv = ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            depth=depth,
            padding_mode=padding_mode,
            res_connection=False,
            activation=activation,
            last_activation=True,
        )
        self.t_linear = nn.Linear(t_embed_size, out_channels)
        self.final_conv = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=_get_padding(kernel_size, 3),
            padding_mode="circular",
        )
        self.act = self.x_conv.act

        if in_channels != out_channels:
            self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_in = x
        x = self.x_conv(x)
        t = self.t_linear(t).reshape(1, x.shape[1], 1, 1, 1)
        x = self.act(x + t)
        x = self.final_conv(x)
        x = x + self.res_conv(x_in)
        return x


class UnetAttentionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mol_embed_channels: int,
        kernel_size: int | tuple[int, ...] = 3,
        depth: int = 2,
        padding_mode: str = "zeros",
        activation: bool = None,
    ):
        super().__init__()

        self.q_conv = ResBlock(
            in_channels=mol_embed_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            depth=depth,
            padding_mode=padding_mode,
            res_connection=True,
            activation=activation,
            last_activation=True,
        )
        self.k_conv = ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            depth=depth,
            padding_mode=padding_mode,
            res_connection=True,
            activation=activation,
            last_activation=True,
        )
        self.v_conv = ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            depth=depth,
            padding_mode=padding_mode,
            res_connection=True,
            activation=activation,
            last_activation=True,
        )
        self.out_conv = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=_get_padding(kernel_size, 3),
            padding_mode="circular",
        )
        self.act = self.q_conv.act

        if in_channels != out_channels:
            self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x: torch.Tensor, mol_embed: torch.Tensor) -> torch.Tensor:
        # x.shape = (n_batch, in_channels, mx, my, mz)
        # mol_embed.shape = (n_batch, mol_embed_channels, nx / 4, ny / 4, nz / 4)

        # mol_embed.shape -> (n_batch, mol_embed_channels, mx, my, mz)
        mol_embed = nn.functional.interpolate(mol_embed, size=x.shape[2:], mode="trilinear", align_corners=False)

        # q.shape = k.shape = v.shape = (n_batch, out_channels, mx, my, mz)
        q = self.q_conv(mol_embed)
        k = self.k_conv(x)
        v = self.v_conv(x)

        qk = q * k  # qk.shape = (n_batch, out_channels, mx, my, mz)
        sh = qk.shape
        qk = qk.reshape(sh[0], sh[1], -1)  # qk.shape = (n_batch, out_channels, mx * my * mz)
        attention = nn.functional.softmax(qk, dim=-1)  # attention.shape = (n_batch, out_channels, mx * my * mz)
        attention = attention.reshape(sh)  # attention.shape = (n_batch, out_channels, mx, my, mz)

        x = self.out_conv(v * attention) + self.res_conv(x)  # x.shape = (n_batch, out_channels, mx, my, mz)

        return x


class DensityDiffusionUNet(nn.Module):

    def __init__(self, out_channels: int, t_embed_size: int, mol_embed_channels: int, device: str | torch.device = "cpu"):
        super().__init__()

        self.act = nn.ReLU()

        self.encoder_blocks = nn.ModuleList(
            [
                # input_shape = (n_batch, 4, nx / 4, ny / 4, nz / 4)
                nn.ModuleList(
                    [
                        UnetResBlock(4, 8, t_embed_size, padding_mode="circular", activation=self.act),
                        UnetAttentionBlock(8, 8, mol_embed_channels, padding_mode="circular", activation=self.act),
                        nn.Identity(),
                    ]
                ),
                nn.ModuleList(
                    [
                        UnetResBlock(8, 8, t_embed_size, padding_mode="circular", activation=self.act),
                        UnetAttentionBlock(8, 8, mol_embed_channels, padding_mode="circular", activation=self.act),
                        # (n_batch, 8, nx / 4, ny / 4, nz / 4) ->  # (n_batch, 8, nx / 8, ny / 8, nz / 8)
                        nn.AvgPool3d(kernel_size=2, stride=2),
                    ]
                ),
                nn.ModuleList(
                    [
                        UnetResBlock(8, 16, t_embed_size, padding_mode="circular", activation=self.act),
                        UnetAttentionBlock(16, 16, mol_embed_channels, padding_mode="circular", activation=self.act),
                        # (n_batch, 16, nx / 8, ny / 8, nz / 8) ->  # (n_batch, 16, nx / 16, ny / 16, nz / 16)
                        nn.AvgPool3d(kernel_size=2, stride=2),
                    ]
                ),
                nn.ModuleList(
                    [
                        UnetResBlock(16, 32, t_embed_size, padding_mode="circular", activation=self.act),
                        UnetAttentionBlock(32, 32, mol_embed_channels, padding_mode="circular", activation=self.act),
                        # (n_batch, 32, nx / 16, ny / 16, nz / 16) ->  # (n_batch, 32, nx / 32, ny / 32, nz / 32)
                        nn.AvgPool3d(kernel_size=2, stride=2),
                    ]
                ),
            ]
        )

        self.middle_block = ResBlock(32, 32, depth=3, padding_mode="circular", activation=self.act)
        # (n_batch, 32, nx / 32, ny / 32, nz / 32)

        self.decoder_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        UnetResBlock(64, 16, t_embed_size, padding_mode="circular", activation=self.act),
                        UnetAttentionBlock(16, 16, mol_embed_channels, padding_mode="circular", activation=self.act),
                        # (n_batch, 16, nx / 32, ny / 32, nz / 32) -> (n_batch, 16, nx / 16, ny / 16, nz / 16)
                        nn.Upsample(scale_factor=2, mode="trilinear"),
                    ]
                ),
                nn.ModuleList(
                    [
                        UnetResBlock(32, 8, t_embed_size, padding_mode="circular", activation=self.act),
                        UnetAttentionBlock(8, 8, mol_embed_channels, padding_mode="circular", activation=self.act),
                        # (n_batch, 8, nx / 16, ny / 16, nz / 16) -> (n_batch, 16, nx / 8, ny / 8, nz / 8)
                        nn.Upsample(scale_factor=2, mode="trilinear"),
                    ]
                ),
                nn.ModuleList(
                    [
                        UnetResBlock(16, 8, t_embed_size, padding_mode="circular", activation=self.act),
                        UnetAttentionBlock(8, 8, mol_embed_channels, padding_mode="circular", activation=self.act),
                        # (n_batch, 8, nx / 8, ny / 8, nz / 8) -> (n_batch, 8, nx / 4, ny / 4, nz / 4)
                        nn.Upsample(scale_factor=2, mode="trilinear"),
                    ]
                ),
                nn.ModuleList(
                    [
                        UnetResBlock(16, 4, t_embed_size, padding_mode="circular", activation=self.act),
                        UnetAttentionBlock(4, 4, mol_embed_channels, padding_mode="circular", activation=self.act),
                        nn.Identity(),
                    ]
                ),
            ]
        )

        self.conv_final = ResBlock(4, out_channels, depth=2, padding_mode="circular", activation=self.act, last_activation=False)

        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor, mol_embed: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:

        # x.shape = (n_batch, 4, nx, ny, nz)
        # mol_embedding.shape = (n_batch, mol_embed_channels, nx, ny, nz)
        # t_embed.shape = (1, t_embed_size)

        skip_conn = []
        for res_block, attention_block, pool in self.encoder_blocks:
            x = res_block(x, t_embed)
            x = attention_block(x, mol_embed)
            x = pool(x)
            skip_conn.append(x)

        x = self.middle_block(x)

        for res_block, attention_block, upsample in self.decoder_blocks:
            x = torch.cat([x, skip_conn.pop()], dim=1)
            x = res_block(x, t_embed)
            x = attention_block(x, mol_embed)
            x = upsample(x)

        x = self.conv_final(x)

        return x


class TimeEmbedding(nn.Module):

    def __init__(self, embedding_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.SiLU(),
            nn.Linear(4 * embedding_size, 4 * embedding_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MolPotentialEncoder(nn.Module):

    def __init__(self, in_channels: int = 2, out_channels: int = 4, device: str | torch.device = "cpu"):
        super().__init__()

        self.act = nn.ReLU()
        self.encoder = nn.Sequential(
            ResBlock(in_channels, 4, padding_mode="circular", activation=self.act),  # (n_batch, 4, nx, ny, nz)
            nn.AvgPool3d(kernel_size=2, stride=2),  # (n_batch, 4, nx / 2, ny / 2, nz / 2)
            ResBlock(4, 8, padding_mode="circular", activation=self.act),  # (n_batch, 8, nx / 2, ny / 2, nz / 2)
            nn.AvgPool3d(kernel_size=2, stride=2),  # (n_batch, 8, nx / 4, ny / 4, nz / 4)
            ResBlock(8, out_channels, padding_mode="circular", activation=self.act),  # (n_batch, out_channels, nx / 4, ny / 4, nz / 4)
        )

        self.device = device
        self.to(device)

    def forward(self, x):
        return self.encoder(x)


class DensityDiffusion(nn.Module):

    def __init__(self, t_embed_size: int = 32, mol_embed_channels: int = 4, device: str | torch.device = "cpu"):
        super().__init__()

        self.unet = DensityDiffusionUNet(t_embed_size, 4 * t_embed_size, mol_embed_channels)
        self.conv_final = nn.Conv3d(t_embed_size, 4, kernel_size=3, padding=1, padding_mode="circular")
        self.time_embedding = TimeEmbedding(t_embed_size)

        self.to(device)

    def forward(self, x: torch.Tensor, mol_embed: torch.Tensor, t: torch.Tensor):
        # latent.shape = (n_batch, 4, nx, ny, nz)
        # mol_embedding.shape = (n_batch, mol_embed_channels, nx, ny, nz)
        # t.shape = (1, 32)

        t_embed = self.time_embedding(t)  # t_embed.shape = (1, 4 * t_embed_size)
        x = self.unet(x, mol_embed, t_embed)
        x = self.conv_final(x)

        return x
