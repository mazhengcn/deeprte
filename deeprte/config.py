from jaxline import base_config
from ml_collections import config_dict

from deeprte import dataset

N_TRAIN_EXAMPLES = dataset.Split.TRAIN_AND_VALID.num_examples
N_CLASSES = 1000
# Only local/debug parameters are supported out of the box.
# To use the scaled-up hyperparameters, please adapt this script to your
# training setup and set this flag to False
IS_LOCAL = True


def get_training_steps(batch_size, n_epochs):
    return (N_TRAIN_EXAMPLES * n_epochs) // batch_size


def get_config():
    """Return config object for training."""
    use_debug_settings = IS_LOCAL
    config = base_config.get_base_config()

    # Experiment config.
    local_batch_size = 2
    # Modify this to adapt to your custom distributed learning setup
    num_devices = 1
    config.train_batch_size = local_batch_size * num_devices
    config.n_epochs = 110

    def _default_or_debug(default_value, debug_value):
        return debug_value if use_debug_settings else default_value

    n_train_examples = N_TRAIN_EXAMPLES
    num_classes = N_CLASSES

    config.experiment_kwargs = config_dict.ConfigDict(
        dict(
            config=dict(
                optimizer=dict(
                    base_lr=5e-4,
                    max_norm=10.0,  # < 0 to turn off.
                    schedule_type="constant_cosine",
                    step_decay_kwargs=dict(
                        decay_boundaries=[0.5, 0.8, 0.95],
                        decay_rate=0.1,
                    ),
                    optimizer="adam",
                    # Optimizer-specific kwargs:
                    adam_kwargs=dict(
                        b1=0.9,
                        b2=0.999,
                        eps=1e-8,
                    ),
                ),
                # Don't specify output_channels - it's not used for
                # classifiers.
                model=dict(
                    operator_kwargs=dict(
                        green_function=dict(
                            prep_type="pixels",
                            # Channels for conv/conv1x1 preprocessing:
                            num_channels=64,
                            # -------------------------
                            # Position encoding arguments:
                            # -------------------------
                            position_encoding_type="fourier",
                            concat_or_add_pos="concat",
                            spatial_downsample=1,
                            # If >0, project position to this size:
                            project_pos_dim=-1,
                            trainable_position_encoding_kwargs=dict(
                                num_channels=258,
                                init_scale=0.02,
                            ),
                            fourier_position_encoding_kwargs=dict(
                                num_bands=64,
                                max_resolution=(224, 224),
                                sine_only=False,
                                concat_pos=True,
                            ),
                        ),
                        coefficient_net=dict(
                            num_self_attends_per_block=_default_or_debug(6, 2),
                            # Weights won't be shared if num_blocks
                            # is set to 1.
                            num_blocks=_default_or_debug(8, 2),
                            z_index_dim=512,
                            num_z_channels=1024,
                            num_cross_attend_heads=1,
                            num_self_attend_heads=8,
                            cross_attend_widening_factor=1,
                            self_attend_widening_factor=1,
                            dropout_prob=0.0,
                            # Position encoding for the latent array.
                            z_pos_enc_init_scale=0.02,
                            cross_attention_shape_for_attn="kv",
                            use_query_residual=True,
                        ),
                        decoder=dict(
                            num_z_channels=1024,
                            use_query_residual=True,
                            # Position encoding for the output logits.
                            position_encoding_type="trainable",
                            trainable_position_encoding_kwargs=dict(
                                num_channels=1024,
                                init_scale=0.02,
                            ),
                        ),
                    ),
                ),
                training=dict(
                    images_per_epoch=n_train_examples,
                    label_smoothing=0.1,
                    n_epochs=config.get_oneway_ref("n_epochs"),
                    batch_size=config.get_oneway_ref("train_batch_size"),
                ),
                data=dict(
                    num_classes=num_classes,
                    # Run on smaller images to debug.
                    im_dim=_default_or_debug(224, 32),
                    augmentation=dict(
                        # Typical randaug params:
                        # num_layers in [1, 3]
                        # magnitude in [5, 30]
                        # Set randaugment to None to disable.
                        randaugment=dict(num_layers=4, magnitude=5),
                        cutmix=True,
                        # Mixup alpha should be in [0, 1].
                        # Set to None to disable.
                        mixup_alpha=0.2,
                    ),
                ),
                evaluation=dict(
                    subset="test",
                    batch_size=2,
                ),
            )
        )
    )

    # Training loop config.
    config.training_steps = get_training_steps(
        config.get_oneway_ref("train_batch_size"),
        config.get_oneway_ref("n_epochs"),
    )
    config.log_train_data_interval = 60
    config.log_tensors_interval = 60
    config.save_checkpoint_interval = 300
    config.eval_specific_checkpoint_dir = ""
    config.best_model_eval_metric = "eval_top_1_acc"
    config.checkpoint_dir = "/tmp/perceiver_imagnet_checkpoints"
    config.train_checkpoint_all_hosts = False

    # Prevents accidentally setting keys that aren't
    # recognized (e.g. in tests).
    config.lock()

    return config
