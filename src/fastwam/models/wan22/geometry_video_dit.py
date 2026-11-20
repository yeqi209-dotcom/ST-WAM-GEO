"""A separate depth-space diffusion expert, using the shared MoT interface."""

from .dino_video_dit import DinoVideoDiT


class GeometryVideoDiT(DinoVideoDiT):
    """Patch normalized depth into tokens and predict dense depth velocity.

    Reuses the generic video DiT implementation, not the DINO encoder or its
    weights. Input/output projections and every transformer block are owned
    by this expert. All frames are noisy; depth is never a policy condition.
    """

    def __init__(
        self, hidden_dim, ffn_dim, text_dim, num_heads, attn_head_dim,
        num_layers, freq_dim=256, eps=1e-6, patch_size=(1, 4, 4),
        use_gradient_checkpointing=True,
    ):
        super().__init__(
            hidden_dim=hidden_dim, dino_dim=1, ffn_dim=ffn_dim,
            text_dim=text_dim, freq_dim=freq_dim, eps=eps,
            num_heads=num_heads, attn_head_dim=attn_head_dim,
            num_layers=num_layers, video_attention_mask_mode="bidirectional",
            use_gradient_checkpointing=use_gradient_checkpointing,
            latent_patch_size=tuple(patch_size), latent_patch_mode="flat",
            output_patch_space="dense",
        )
        self.output_channels = 1
        self.fuse_vae_embedding_in_latents = False

    def pre_dit(self, x, timestep, context, context_mask=None,
                fuse_vae_embedding_in_latents=False):
        if x.ndim != 5 or x.shape[1] != 1:
            raise ValueError("Geometry expert input must be [B,1,T,H,W].")
        if fuse_vae_embedding_in_latents:
            raise ValueError("Geometry expert has no clean first-frame depth condition.")
        return super().pre_dit(
            x=x, timestep=timestep, context=context, context_mask=context_mask,
            fuse_vae_embedding_in_latents=False,
        )
