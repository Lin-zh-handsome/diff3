"""Third-chapter TM-DP policy: light temporal shift and motion excitation.

Only changes the observation encoder. The original CNN-DP action U-Net,
noise scheduler, diffusion objective, and receding-horizon execution are kept.
"""
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
    DiffusionUnetHybridImagePolicy,
)
from diffusion_policy.model.vision.temporal_motion_encoder import (
    TemporalMotionObsEncoder,
)


class TMDiffusionUnetHybridImagePolicy(DiffusionUnetHybridImagePolicy):
    def __init__(
        self,
        use_tsm=False,
        use_motion_excitation=False,
        use_motion_condition_fusion=None,
        tsm_channels=(128, 256),
        me_channels=(256,),
        tsm_fold_div=8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_tsm = bool(use_tsm)
        self.use_motion_excitation = bool(use_motion_excitation)
        self.use_motion_condition_fusion = (
            self.use_motion_excitation if use_motion_condition_fusion is None
            else bool(use_motion_condition_fusion)
        )
        if self.use_tsm or self.use_motion_excitation or self.use_motion_condition_fusion:
            if not self.obs_as_global_cond:
                raise ValueError('TM-DP currently requires obs_as_global_cond=True')
            self.obs_encoder = TemporalMotionObsEncoder(
                self.obs_encoder,
                n_obs_steps=self.n_obs_steps,
                feature_dim=self.obs_feature_dim,
                use_tsm=self.use_tsm,
                use_motion_excitation=self.use_motion_excitation,
                use_motion_condition_fusion=self.use_motion_condition_fusion,
                tsm_channels=tsm_channels,
                me_channels=me_channels,
                fold_div=tsm_fold_div,
            )
            print('TM-DP: TSM=%s, MotionExcitation=%s, MotionFusion=%s, temporal blocks=%d' % (
                self.use_tsm,
                self.use_motion_excitation,
                self.use_motion_condition_fusion,
                self.obs_encoder.installed_blocks,
            ))
