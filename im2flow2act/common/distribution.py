import torch
from torch.distributions import Categorical, OneHotCategorical, kl_divergence
from torch.distributions import Normal as TorchNormal
from torch.distributions import Beta as TorchBeta
from torch.distributions import Distribution as TorchDistribution
from torch.distributions import Bernoulli as TorchBernoulli
from torch.distributions import Independent as TorchIndependent
from torch.distributions.utils import _sum_rightmost
import numpy as np
from collections import OrderedDict
import torch.distributions as D
import torch as th

MAX_LOGPROB = 100
MIN_LOGPROB = -100


class Distribution(TorchDistribution):
    def sample_and_logprob(self):
        s = self.sample()
        log_p = self.log_prob(s)
        return s, log_p

    def rsample_and_logprob(self):
        s = self.rsample()
        log_p = self.log_prob(s)
        return s, log_p

    def mle_estimate(self):
        return self.mean

    def get_diagnostics(self):
        return {}


class Independent(Distribution, TorchIndependent):
    def get_diagnostics(self):
        return self.base_dist.get_diagnostics()


class TorchDistributionWrapper(Distribution):
    def __init__(self, distribution: TorchDistribution):
        self.distribution = distribution

    @property
    def batch_shape(self):
        return self.distribution.batch_shape

    @property
    def event_shape(self):
        return self.distribution.event_shape

    @property
    def arg_constraints(self):
        return self.distribution.arg_constraints

    @property
    def support(self):
        return self.distribution.support

    @property
    def mean(self):
        return self.distribution.mean

    @property
    def variance(self):
        return self.distribution.variance

    @property
    def stddev(self):
        return self.distribution.stddev

    def sample(self, sample_size=torch.Size()):
        return self.distribution.sample(sample_shape=sample_size)

    def rsample(self, sample_size=torch.Size()):
        return self.distribution.rsample(sample_shape=sample_size)

    def log_prob(self, value):
        return self.distribution.log_prob(value)

    def cdf(self, value):
        return self.distribution.cdf(value)

    def icdf(self, value):
        return self.distribution.icdf(value)

    def enumerate_support(self, expand=True):
        return self.distribution.enumerate_support(expand=expand)

    def entropy(self):
        return self.distribution.entropy()

    def perplexity(self):
        return self.distribution.perplexity()

    def __repr__(self):
        return "Wrapped " + self.distribution.__repr__()


class MultivariateDiagonalNormal(TorchDistributionWrapper):
    from torch.distributions import constraints

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}

    def __init__(self, loc, scale_diag, reinterpreted_batch_ndims=1):
        dist = Independent(
            TorchNormal(loc, scale_diag),
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
        )
        super().__init__(dist)

    def __repr__(self):
        return self.distribution.base_dist.__repr__()


@torch.distributions.kl.register_kl(TorchDistributionWrapper, TorchDistributionWrapper)
def _kl_mv_diag_normal_mv_diag_normal(p, q):
    return kl_divergence(p.distribution, q.distribution)


# Independent RV KL handling - https://github.com/pytorch/pytorch/issues/13545


@torch.distributions.kl.register_kl(TorchIndependent, TorchIndependent)
def _kl_independent_independent(p, q):
    if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
        raise NotImplementedError
    result = kl_divergence(p.base_dist, q.base_dist)
    return _sum_rightmost(result, p.reinterpreted_batch_ndims)


class ScaledMultivariateDiagonalNormal(TorchDistributionWrapper):
    from torch.distributions import constraints

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}

    def __init__(
        self, loc, scale_diag, scaling_constant=1, reinterpreted_batch_ndims=1
    ):
        dist = Independent(
            TorchNormal(loc, scale_diag),
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
        )
        super().__init__(dist)
        self.scaling_constant = scaling_constant
        self.eps = 1e-3

    def log_prob(self, value):
        value = value * self.scaling_constant
        log_prob = self.distribution.log_prob(value)
        return torch.clamp(log_prob, min=MIN_LOGPROB, max=MAX_LOGPROB)

    def sample(self):
        value = self.rsample()

        return value.detach()

    def rsample(self):
        value = self.distribution.rsample()
        value = torch.clamp(
            value, -self.scaling_constant + self.eps, self.scaling_constant - self.eps
        )
        value = value / self.scaling_constant

        return value

    def __repr__(self):
        return self.distribution.base_dist.__repr__()


class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)
    Note: this is not very numerically stable.
    """

    def __init__(self, normal_mean, normal_std, epsilon=1e-6, device=None):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = MultivariateDiagonalNormal(normal_mean, normal_std)
        self.epsilon = epsilon
        self.device = device

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def _log_prob_from_pre_tanh(self, pre_tanh_value):
        """
        Adapted from
        https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L73
        This formula is mathematically equivalent to log(1 - tanh(x)^2).
        Derivation:
        log(1 - tanh(x)^2)
         = log(sech(x)^2)
         = 2 * log(sech(x))
         = 2 * log(2e^-x / (e^-2x + 1))
         = 2 * (log(2) - x - log(e^-2x + 1))
         = 2 * (log(2) - x - softplus(-2x))
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        log_prob = self.normal.log_prob(pre_tanh_value)
        correction = -2.0 * (
            torch.tensor(np.log([2.0]), dtype=torch.float32, device=self.device)
            - pre_tanh_value
            - torch.nn.functional.softplus(-2.0 * pre_tanh_value)
        ).sum(dim=1)
        return log_prob + correction

    # def log_prob(self, value, pre_tanh_value=None):
    #     if pre_tanh_value is None:
    #         pre_tanh_value = TanhBijector.inverse(value)
    #     return self._log_prob_from_pre_tanh(pre_tanh_value)
    def log_prob(self, value, pre_tanh_value=None):
        if pre_tanh_value is None:
            value = torch.clamp(value, -0.999, 0.999)
            pre_tanh_value = TanhBijector.inverse(value)
        log_prob = self.normal.log_prob(pre_tanh_value)
        log_prob -= torch.sum(th.log(1 - value**2 + self.epsilon), dim=1)
        return log_prob

    def rsample_with_pretanh(self):
        # z = (self.normal_mean + self.normal_std *
        #      MultivariateDiagonalNormal(torch.zeros(self.normal_mean.size(), device=self.device),
        #                                 torch.ones(self.normal_std.size(), device=self.device)).sample())
        z = self.normal.rsample()
        return torch.tanh(z), z

    def sample(self):
        """
        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        value, pre_tanh_value = self.rsample_with_pretanh()
        return value.detach()

    def rsample(self):
        """
        Sampling in the reparameterization case.
        """
        value, pre_tanh_value = self.rsample_with_pretanh()
        return value

    def sample_and_logprob(self):
        value, pre_tanh_value = self.rsample_with_pretanh()
        value, pre_tanh_value = value.detach(), pre_tanh_value.detach()
        log_p = self.log_prob(value, pre_tanh_value)
        return value, log_p

    def rsample_and_logprob(self):
        value, pre_tanh_value = self.rsample_with_pretanh()
        log_p = self.log_prob(value, pre_tanh_value)
        return value, log_p

    @property
    def mean(self):
        return torch.tanh(self.normal_mean)


### copy from stable baseline3
class TanhBijector(object):
    """
    Bijective transformation of a probability distribution
    using a squashing function (tanh)
    TODO: use Pyro instead (https://pyro.ai/)
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, epsilon: float = 1e-6):
        super(TanhBijector, self).__init__()
        self.epsilon = epsilon

    @staticmethod
    def forward(x: th.Tensor) -> th.Tensor:
        return th.tanh(x)

    @staticmethod
    def atanh(x: th.Tensor) -> th.Tensor:
        """
        Inverse of Tanh
        Taken from Pyro: https://github.com/pyro-ppl/pyro
        0.5 * torch.log((1 + x ) / (1 - x))
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y: th.Tensor) -> th.Tensor:
        """
        Inverse tanh.
        :param y:
        :return:
        """
        eps = th.finfo(y.dtype).eps
        # Clip the action to avoid NaN
        return TanhBijector.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))

    def log_prob_correction(self, x: th.Tensor) -> th.Tensor:
        # Squash correction (from original SAC implementation)
        return th.log(1.0 - th.tanh(x) ** 2 + self.epsilon)


class GMMDistribution(Distribution):
    
    def __init__(self,means,scales,logits):
        self.means  = means
        self.scales = scales
        self.logits =logits

        component_distribution = D.Normal(loc=means, scale=scales)
        component_distribution = D.Independent(component_distribution, 1)

        # unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = D.Categorical(logits=logits)

        self.dist = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )
        
    def log_prob(self,value):
        
        return self.dist.log_prob(value)
    
    def sample(self):
        
        return self.dist.sample()

    def __repr__(self):
        s = "GaussianMixture(means=%s,stds=%s, weights=%s)"
        return s % (self.means, self.scales, torch.softmax(self.logits,dim=1))