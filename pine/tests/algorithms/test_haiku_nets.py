import jax
import haiku as hk
import pytest

import pine.algorithms.haiku_nets as nets


@pytest.mark.parametrize("use_v2", [True, False])
def test_residual_conv_block(use_v2):
    def encoder_fn(observations):
        block_cls = nets.ResidualConvBlockV2 if use_v2 else nets.ResidualConvBlockV1
        return block_cls(channels=16, stride=2, use_projection=True)(
            observations
        )

    def decoder_fn(states):
        block_cls = (
            nets.ResidualTransposedConvBlockV2
            if use_v2
            else nets.ResidualTransposedConvBlockV1
        )
        return block_cls(channels=3, stride=2, use_projection=True)(states)

    encoder_model = hk.without_apply_rng(hk.transform(encoder_fn))
    decoder_model = hk.without_apply_rng(hk.transform(decoder_fn))

    rng = jax.random.PRNGKey(0)
    obs = jax.random.uniform(rng, (1, 96, 96, 3))
    encoder_params = encoder_model.init(rng, obs)
    state = encoder_model.apply(encoder_params, obs)
    decoder_params = decoder_model.init(rng, state)
    out = decoder_model.apply(decoder_params, state)
    assert out.shape == obs.shape


@pytest.mark.parametrize("use_v2", [True, False])
def test_reconstruction(use_v2):
    def recon_model_fn(observations):
        encoder = nets.EZStateEncoder(channels=16, use_v2=use_v2)
        decoder = nets.EZStateDecoder(channels=16, use_v2=use_v2)
        states = encoder(observations)
        return decoder(states)

    recon_model = hk.without_apply_rng(hk.transform(recon_model_fn))

    rng = jax.random.PRNGKey(0)
    obs = jax.random.uniform(rng, (1, 96, 96, 3))
    recon_model_params = recon_model.init(rng, obs)
    recon_out = recon_model.apply(recon_model_params, obs)
    assert recon_out.shape == obs.shape
