"""Minimal RMA modules kept for FAME *inference*.

Rapid Motor Adaptation (Kumar et al., RSS 2021) trains an env-factor encoder
that maps privileged factors ``e_t`` to a compact latent ``z_t``; the policy is
conditioned on ``(obs, z_t)``. At deploy time we run the encoder directly on the
``e_t`` we can observe (upper-body joint positions + wrist forces).

Only the encoder (and the ``e_t`` spec that documents its 21-dim layout) is
needed to run the policy. The training-only pieces — the decoder
(reconstruction loss), the phase-2 adaptation CNN, and the Isaac-Gym e_t/force
builders — have been removed from this package.
"""

from .env_factor_encoder import EnvFactorEncoder, EnvFactorEncoderCfg
from .env_factor_spec import DEFAULT_ET_SPEC, RmaEtSpec, UPPER_BODY_JOINT_NAMES

__all__ = [
    "EnvFactorEncoder",
    "EnvFactorEncoderCfg",
    "DEFAULT_ET_SPEC",
    "RmaEtSpec",
    "UPPER_BODY_JOINT_NAMES",
]
