import functools
import os

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from tapnet import tapir_model
from tapnet.utils import model_utils


def construct_initial_causal_state(num_points, num_resolutions):
    """Construct initial causal state."""
    value_shapes = {
        "tapir/~/pips_mlp_mixer/block_1_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_1_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_2_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_2_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_3_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_3_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_4_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_4_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_5_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_5_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_6_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_6_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_7_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_7_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_8_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_8_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_9_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_9_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_10_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_10_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_11_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_11_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_causal_2": (1, num_points, 2, 2048),
    }
    fake_ret = {k: jnp.zeros(v, dtype=jnp.float32) for k, v in value_shapes.items()}
    return [fake_ret] * num_resolutions * 4


def load_checkpoint(checkpoint_path):
    ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
    return ckpt_state["params"], ckpt_state["state"]


def build_online_model_init(frames, points):
    model = tapir_model.TAPIR(
        use_causal_conv=True, bilinear_interp_with_depthwise_conv=False
    )
    feature_grids = model.get_feature_grids(frames, is_training=False)
    features = model.get_query_features(
        frames,
        is_training=False,
        query_points=points,
        feature_grids=feature_grids,
    )
    return features


def build_online_model_predict(frames, features, causal_context):
    """Compute point tracks and occlusions given frames and query points."""
    model = tapir_model.TAPIR(
        use_causal_conv=True, bilinear_interp_with_depthwise_conv=False
    )
    feature_grids = model.get_feature_grids(frames, is_training=False)
    trajectories = model.estimate_trajectories(
        frames.shape[-3:-1],
        is_training=False,
        feature_grids=feature_grids,
        query_features=features,
        query_points_in_video=None,
        query_chunk_size=64,
        causal_context=causal_context,
        get_causal_context=True,
    )
    causal_context = trajectories["causal_context"]
    del trajectories["causal_context"]
    return {k: v[-1] for k, v in trajectories.items()}, causal_context


def sample_random_points(frame_max_idx, height, width, num_points):
    """Sample random points with (time, height, width) order."""
    y = np.random.randint(0, height, (num_points, 1))
    x = np.random.randint(0, width, (num_points, 1))
    t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
    points = np.concatenate((t, y, x), axis=-1).astype(np.int32)  # [num_points, 3]
    return points


def preprocess_frames(frames):
    """Preprocess frames to model inputs.

    Args:
      frames: [num_frames, height, width, 3], [0, 255], np.uint8

    Returns:
      frames: [num_frames, height, width, 3], [-1, 1], np.float32
    """
    frames = frames.astype(np.float32)
    frames = frames / 255 * 2 - 1
    return frames


def postprocess_occlusions(occlusions, expected_dist):
    """Postprocess occlusions to boolean visible flag.

    Args:
      occlusions: [num_points, num_frames], [-inf, inf], np.float32

    Returns:
      visibles: [num_points, num_frames], bool
    """
    pred_occ = jax.nn.sigmoid(occlusions)
    pred_occ = 1 - (1 - pred_occ) * (1 - jax.nn.sigmoid(expected_dist))
    visibles = pred_occ < 0.5  # threshold
    return visibles


def build(num_points, img_size):
    dev_path = os.getenv("DEV_PATH")
    checkpoint_path = os.path.join(
        dev_path, "im2flow2act/tapnet/checkpoints/causal_tapir_checkpoint.npy"
    )
    print("Loaded TAPIR model from", checkpoint_path)
    params, state = load_checkpoint(checkpoint_path)

    print("Creating online tapnet model...")
    online_init = hk.transform_with_state(build_online_model_init)
    online_init_apply = jax.jit(online_init.apply)

    online_predict = hk.transform_with_state(build_online_model_predict)
    online_predict_apply = jax.jit(online_predict.apply)

    rng = jax.random.PRNGKey(42)
    online_init_apply = functools.partial(
        online_init_apply, params=params, state=state, rng=rng
    )
    online_predict_apply = functools.partial(
        online_predict_apply, params=params, state=state, rng=rng
    )
    dummy_frame = np.zeros((*img_size, 3), dtype=np.uint8)
    query_points = jnp.zeros([num_points, 3], dtype=jnp.float32)
    query_features, _ = online_init_apply(
        frames=model_utils.preprocess_frames(dummy_frame[None, None]),
        points=query_points[None, 0:1],
    )
    jax.block_until_ready(query_features)

    query_features, _ = online_init_apply(
        frames=model_utils.preprocess_frames(dummy_frame[None, None]),
        points=query_points[None],
    )
    causal_state = construct_initial_causal_state(
        num_points, len(query_features.resolutions) - 1
    )
    (prediction, causal_state), _ = online_predict_apply(
        frames=model_utils.preprocess_frames(dummy_frame[None, None]),
        features=query_features,
        causal_context=causal_state,
    )

    jax.block_until_ready(prediction["tracks"])
    return online_predict_apply, online_init_apply, online_predict, online_init


def construct_initial_features_and_state(
    query_points, initial_frame, online_init_apply
):
    query_features, _ = online_init_apply(
        frames=preprocess_frames(initial_frame[None, None]),
        points=query_points[None],
    )
    causal_state = construct_initial_causal_state(
        query_points.shape[0], len(query_features.resolutions) - 1
    )
    return query_features, causal_state


def inference(query_features, causal_state, current_frame, online_predict_apply):
    (prediction, causal_state), _ = online_predict_apply(
        frames=preprocess_frames(current_frame[None, None]),
        features=query_features,
        causal_context=causal_state,
    )
    tracks = prediction["tracks"][0]  # (N,1,2)
    occlusions = prediction["occlusion"][0]
    expected_dist = prediction["expected_dist"][0]
    visibles = postprocess_occlusions(occlusions, expected_dist)  # (N,1)
    point_tracking_sequence = np.concatenate(
        [tracks, visibles[:, :, None]], axis=-1
    )  # (N,1,3)
    return point_tracking_sequence, causal_state
