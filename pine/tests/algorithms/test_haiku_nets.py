import jax
import haiku as hk

import pine.algorithms.haiku_nets as nets


def test_residual_conv_block():
    def encoder_fn(observations):
        return nets.ResidualConvBlockV2(channels=16, stride=2, use_projection=True)(
            observations
        )

    def decoder_fn(states):
        return nets.ResidualTransposedConvBlockV2(
            channels=3, stride=2, use_projection=True
        )(states)

    encoder_model = hk.without_apply_rng(hk.transform(encoder_fn))
    decoder_model = hk.without_apply_rng(hk.transform(decoder_fn))

    rng = jax.random.PRNGKey(0)
    obs = jax.random.uniform(rng, (1, 96, 96, 3))
    encoder_params = encoder_model.init(rng, obs)
    state = encoder_model.apply(encoder_params, obs)
    decoder_params = decoder_model.init(rng, state)
    out = decoder_model.apply(decoder_params, state)
    assert out.shape == obs.shape


def test_reconstruction():
    def recon_model_fn(observations):
        encoder = nets.EZStateEncoder(channels=16, use_v2=True)
        decoder = nets.EZStateDecoder(channels=16, use_v2=True)
        states = encoder(observations)
        return decoder(states)

    recon_model = hk.without_apply_rng(hk.transform(recon_model_fn))

    rng = jax.random.PRNGKey(0)
    obs = jax.random.uniform(rng, (1, 96, 96, 3))
    recon_model_params = recon_model.init(rng, obs)
    recon_out = recon_model.apply(recon_model_params, obs)
    assert recon_out.shape == obs.shape
