import numpy as np
import torch
from torch import nn

LOG_STD_MAX = 2
LOG_STD_MIN = -20

from typing import List, Type

import torch
import torch.nn as nn
from im2flow2act.diffusion_policy.model.distribution import (
    GMMDistribution,
    MultivariateDiagonalNormal,
)
from torch.distributions.categorical import Categorical
from torch.nn.utils import spectral_norm


def add_sn(m):
    for name, c in m.named_children():
        m.add_module(name, add_sn(c))

    if isinstance(m, (nn.Conv2d, nn.Linear)):
        return nn.utils.spectral_norm(m)
    else:
        return m


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    use_batch_norm: bool = False,
    use_group_norm: bool = False,
    use_spectral_norm: bool = False,
    squash_output: bool = False,
    sigmoid_output: bool = False,
    dropout_prob: float = 0,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.
    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        if use_spectral_norm:
            modules = [
                spectral_norm(nn.Linear(input_dim, net_arch[0])),
                activation_fn(),
            ]
        else:
            modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        if use_spectral_norm:
            modules.append(spectral_norm(nn.Linear(net_arch[idx], net_arch[idx + 1])))
        else:
            modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        if use_batch_norm:
            modules.append(nn.BatchNorm1d(num_features=net_arch[idx]))
        if use_group_norm:
            modules.append(
                nn.GroupNorm(num_groups=net_arch[idx] // 16, num_channels=net_arch[idx])
            )
        modules.append(activation_fn())
        if dropout_prob > 0:
            modules.append(nn.Dropout(dropout_prob))

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    if sigmoid_output:
        modules.append(nn.Sigmoid())
    return modules


class MLPCategoricalActor(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        net_arch: List[int],
        use_batch_norm=False,
        use_group_norm=False,
        use_spectral_norm=False,
    ):
        super().__init__()
        self.logits_net = nn.Sequential(
            *create_mlp(
                input_dim=in_size,
                output_dim=out_size,
                net_arch=net_arch,
                use_batch_norm=use_batch_norm,
                use_group_norm=use_group_norm,
                use_spectral_norm=use_spectral_norm,
            )
        )

    def forward(self, obs, act=None):
        distribution = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(distribution, act)
        return distribution, logp_a

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class Mlp(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        net_arch,
        use_batch_norm=False,
        use_group_norm=False,
        use_spectral_norm=False,
        sigmoid_output=False,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            *create_mlp(
                input_dim=in_size,
                output_dim=out_size,
                net_arch=net_arch,
                use_batch_norm=use_batch_norm,
                use_group_norm=use_group_norm,
                use_spectral_norm=use_spectral_norm,
                sigmoid_output=sigmoid_output,
            )
        )

    def forward(self, x):
        x = self.net(x)
        return x


class GaussianMlp(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        net_arch=[128, 128],
        use_batch_norm=False,
        use_group_norm=False,
        dropout_prob=0,
        latent_drop_prob=0,
    ):
        super().__init__()

        latent_policy = create_mlp(
            input_dim=in_size,
            output_dim=-1,
            net_arch=net_arch,
            use_batch_norm=use_batch_norm,
            use_group_norm=use_group_norm,
            dropout_prob=dropout_prob,
        )
        self.latent_drop_prob = latent_drop_prob
        self.latent_policy = nn.Sequential(*latent_policy)
        self.drop_out = nn.Dropout(self.latent_drop_prob)
        self.mu = nn.Linear(net_arch[-1], out_size)
        self.log_std = nn.Linear(net_arch[-1], out_size)

    def forward(self, input):
        policy_latent = self.latent_policy(input)
        if self.latent_drop_prob > 0:
            policy_latent = self.drop_out(policy_latent)
        mu = self.mu(policy_latent)
        log_std = self.log_std(policy_latent)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return MultivariateDiagonalNormal(mu, log_std.exp())


class MixinGaussianBcPolicy(nn.Module):
    def __init__(
        self,
        vision_encoder,
        state_encoder,
        mix_net,
        action_dim,
        vision_only=False,
        bc_net_arch=[128, 128],
        use_batch_norm=False,
        use_group_norm=False,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.state_encoder = state_encoder
        self.mix_net = mix_net
        self.vision_only = vision_only

        self.bc_policy = GaussianMlp(
            in_size=self.mix_net.net[-1].out_features,
            out_size=action_dim,
            net_arch=bc_net_arch,
            use_batch_norm=use_batch_norm,
            use_group_norm=use_group_norm,
        )

    def forward(self, image, state):
        image_encoding = self.vision_encoder(image)
        if self.vision_only:
            mixin_encoding = image_encoding
        elif self.state_encoder is None:
            mixin_encoding = torch.cat([image_encoding, state], dim=-1)
        else:
            state_encoding = self.state_encoder(state)
            mixin_encoding = torch.cat([image_encoding, state_encoding], dim=-1)

        mix_out = self.mix_net(mixin_encoding)

        action_distribution = self.bc_policy(mix_out)

        return action_distribution


class GMMMlp(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        n_mode,
        net_arch=[128, 128],
        use_batch_norm=False,
        use_group_norm=False,
    ):
        super().__init__()
        latent_policy = create_mlp(
            input_dim=in_size,
            output_dim=-1,
            net_arch=net_arch,
            use_batch_norm=use_batch_norm,
            use_group_norm=use_group_norm,
        )
        self.n_mode = n_mode
        self.out_size = out_size
        self.latent_policy = nn.Sequential(*latent_policy)
        self.mu = nn.Linear(net_arch[-1], out_size * self.n_mode)
        self.log_std = nn.Linear(net_arch[-1], out_size * self.n_mode)
        self.weights = nn.Linear(net_arch[-1], self.n_mode)

    def forward(self, input):
        policy_latent = self.latent_policy(input)
        mu = self.mu(policy_latent)
        log_std = self.log_std(policy_latent)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        logits = self.weights(policy_latent)

        gmm_mu = torch.stack(mu.split(self.out_size, dim=1), dim=1)
        gmm_log_std = torch.stack(log_std.split(self.out_size, dim=1), dim=1)

        return GMMDistribution(gmm_mu, gmm_log_std.exp(), logits)


class LSTMGenerator(nn.Module):
    """An LSTM based generator. It expects a sequence of noise vectors as input.
    Args:
        in_dim: Input noise dimensionality
        out_dim: Output dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms
    Input: noise of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, out_dim)
    """

    def __init__(
        self,
        in_size,
        out_size,
        hidden_dim=256,
        n_layers=1,
        net_arch=[256, 256],
        use_batch_norm=False,
        use_group_norm=False,
        use_spectral_norm=True,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.out_size = out_size

        self.lstm = nn.LSTM(in_size, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Sequential(
            Mlp(
                hidden_dim,
                out_size,
                net_arch,
                use_batch_norm,
                use_group_norm=use_group_norm,
                use_spectral_norm=use_spectral_norm,
            )
        )

    def forward(self, input, prototype):
        """
        input: bxTxf
        prototype: bxd
        """
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()
        hidden = (h_0, c_0)
        outputs = []
        for i in range(seq_len):
            recurrent_features, hidden = self.lstm(
                torch.cat([input[:, i], prototype], dim=1).unsqueeze(1), hidden
            )
            out = self.linear(recurrent_features.squeeze(1))
            outputs.append(out)
        outputs = torch.stack(outputs, dim=1)
        return outputs
