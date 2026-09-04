# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

__all__ = ['SCAILModel']

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply_ref(x, freqs, **kwargs):
    rope_key = kwargs.get("rope_key", "ref")
    f = kwargs.get("rope_ref_T", {}).get(rope_key, 1)
    h = kwargs["rope_H"]
    w = kwargs["rope_W"]
    shift_f = kwargs["rope_T_shift"][rope_key]
    shift_h = kwargs["rope_H_shift"][rope_key]
    shift_w = kwargs["rope_W_shift"][rope_key]

    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    seq_len = f * h * w
    assert seq_len == x.size(1)
    x_i = torch.view_as_complex(x[:, :seq_len].to(torch.float64).reshape(
        x.size(0), seq_len, n, -1, 2))
    freqs_i = torch.cat([
        freqs[0][shift_f:shift_f+f].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs[1][shift_h:shift_h+h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs[2][shift_w:shift_w+w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(seq_len, 1, -1)
    x_i = torch.view_as_real(x_i * freqs_i).flatten(3)
    return torch.cat([x_i, x[:, seq_len:]], dim=1).float()

@amp.autocast(enabled=False)
def rope_apply_additional_ref(x, freqs, **kwargs):
    kwargs = dict(kwargs)
    kwargs["rope_key"] = "additional_ref"
    return rope_apply_ref(x, freqs, **kwargs)

@amp.autocast(enabled=False)
def rope_apply_video(x, freqs, **kwargs):
    f = kwargs["rope_T"]
    h = kwargs["rope_H"]
    w = kwargs["rope_W"]
    shift_f = kwargs["rope_T_shift"]["video"]
    shift_h = kwargs["rope_H_shift"]["video"]
    shift_w = kwargs["rope_W_shift"]["video"]

    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    seq_len = f * h * w
    assert seq_len == x.size(1)

    # precompute multipliers
    x_i = torch.view_as_complex(x[:, :seq_len].to(torch.float64).reshape(
        x.size(0), seq_len, n, -1, 2))
    freqs_i = torch.cat([
        freqs[0][shift_f:shift_f+f].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs[1][shift_h:shift_h+h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs[2][shift_w:shift_w+w].view(1, 1, w, -1).expand(f, h, w, -1)
    ],
                        dim=-1).reshape(seq_len, 1, -1)

    # apply rotary embedding
    x_i = torch.view_as_real(x_i * freqs_i).flatten(3)
    x_i = torch.cat([x_i, x[:, seq_len:]], dim=1)
    return x_i.float()

@amp.autocast(enabled=False)
def rope_apply_pose(x, freqs, **kwargs):
    f = kwargs["rope_T"]
    h = kwargs["rope_H"]
    w = kwargs["rope_W"]
    shift_f = kwargs["rope_T_shift"]["pose"]
    shift_h = kwargs["rope_H_shift"]["pose"]
    shift_w = kwargs["rope_W_shift"]["pose"]

    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    seq_len = f * (h // 2) * (w // 2) # downsampled
    assert seq_len == x.size(1)
    x_i = torch.view_as_complex(x[:, :seq_len].to(torch.float64).reshape(
        x.size(0), seq_len, n, -1, 2))
    freqs_i = torch.cat([
        freqs[0][shift_f:shift_f+f].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs[1][shift_h:shift_h+h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs[2][shift_w:shift_w+w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1) # T H W D

    assert shift_w + w <= freqs[2].size(0), f"{shift_w + w} > {freqs[2].size(0)}"
    freqs_i_real = F.avg_pool2d(
        freqs_i.real.permute(0, 3, 1, 2), kernel_size=2, stride=2
    ).permute(0, 2, 3, 1)
    freqs_i_imag = F.avg_pool2d(
        freqs_i.imag.permute(0, 3, 1, 2), kernel_size=2, stride=2
    ).permute(0, 2, 3, 1)
    freqs_i = torch.complex(freqs_i_real, freqs_i_imag).reshape(seq_len, 1, -1)
    x_i = torch.view_as_real(x_i * freqs_i).flatten(3)
    return torch.cat([x_i, x[:, seq_len:]], dim=1).float()

def rope_apply_scail(x, **kwargs):
    """
    x: [b, s, n, d]
    """
    ref_length = kwargs["ref_length"]
    video_length = kwargs["seq_length"]
    pose_length = kwargs["pose_length"]
    additional_ref_length = kwargs.get("additional_ref_length", 0)

    additional_ref_start = 0
    additional_ref_end = additional_ref_length
    ref_start = additional_ref_end
    ref_end = ref_start + ref_length
    video_start = ref_end
    video_end = video_start + video_length
    pose_start = video_end
    pose_end = pose_start + pose_length

    chunks = []
    if additional_ref_length > 0:
        x_additional_ref = x[:, additional_ref_start:additional_ref_end]
        chunks.append(rope_apply_additional_ref(x_additional_ref, **kwargs))

    x_ref = x[:, ref_start:ref_end]
    x_video = x[:, video_start:video_end]
    x_pose = x[:, pose_start:pose_end]
    chunks.extend([
        rope_apply_ref(x_ref, **kwargs),
        rope_apply_video(x_video, **kwargs),
        rope_apply_pose(x_pose, **kwargs),
    ])

    expected_length = additional_ref_length + ref_length + video_length + pose_length
    assert expected_length == x.size(1), f"RoPE sequence split mismatch: {expected_length} != {x.size(1)}"

    return torch.cat(chunks, dim=1)

class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)

class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, rope_apply_func, **kwargs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply_func(q),
            k=rope_apply_func(k),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        context,
        context_lens,
        **kwargs, # contains rope_apply_func
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, **kwargs)
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(
                torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280))

    def forward(self, image_embeds):
        if hasattr(self, 'emb_pos'):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens

from einops import rearrange
from functools import partial, reduce
from operator import mul

class SCAIL2Model(ModelMixin, ConfigMixin):
    r"""
    SCAIL2 diffusion backbone.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 mask_dim=28,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 pose_rope_shift=[0,0,120], # shift in (t, h, w) for pose rope embedding
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video) or 'vace'
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'flf2v', 'vace']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.mask_dim = mask_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.pose_rope_shift = pose_rope_shift
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        self.patch_embedding_pose = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        self.patch_embedding_mask = nn.Conv3d(
            mask_dim, dim, kernel_size=patch_size, stride=patch_size)
        
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(8192, d - 4 * (d // 6)),
            rope_params(8192, 2 * (d // 6)),
            rope_params(8192, 2 * (d // 6))
        ],
                               dim=1)
        self.hidden_size_head = d

        if model_type == 'i2v' or model_type == 'flf2v':
            self.img_emb = MLPProj(1280, dim, flf_pos_emb=model_type == 'flf2v')

        # initialize weights
        self.init_weights()

    def apply_i2v_ones_masks(self, inputs: torch.Tensor, mask_dim: int = 4):
        b, d, t, h, w= inputs.shape
        mask = torch.ones(b, mask_dim, t, h, w, device=inputs.device, dtype=inputs.dtype)
        inputs = torch.concat([inputs, mask], dim=1)
        return inputs
    
    def apply_i2v_zeros_masks(self, inputs: torch.Tensor, mask_dim: int = 4):
        b, d, t, h, w= inputs.shape
        mask = torch.zeros(b, mask_dim, t, h, w, device=inputs.device, dtype=inputs.dtype)
        inputs = torch.concat([inputs, mask], dim=1)
        return inputs

    def merge_list_of_tensors_to_batch(self, inputs: list[torch.Tensor]):
        return torch.cat([u.unsqueeze(0) for u in inputs], dim=0)

    def forward(
        self,
        x: list[torch.Tensor],
        pose_latents: list[torch.Tensor], 
        driving_masks: list[torch.Tensor],
        ref_latents: list[torch.Tensor], 
        ref_masks: list[torch.Tensor],
        t,
        context,
        seq_len,
        replace_flag: bool,
        history_mask: torch.Tensor=None,
        clip_fea=None,
        additional_ref_latents: list[torch.Tensor]=None,
        additional_ref_masks: list[torch.Tensor]=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            ref_latents (list[Tensor]):
                list of reference latents, each with shape [C_in, 1, H, W]
            ref_masks (list[Tensor]):
                list of reference mask latents, each with shape [C_in, 1 + F, H, W]
            pose_latents (list[Tensor]):
                list of downsampled pose video latents, each with shape [C_in, F, H / 2, W / 2]
            driving_masks (list[Tensor]):
                list of downsampled driving mask latents, each with shape [C_mask, F, H / 2, W / 2]
            history_mask (list[Tensor]):
                list of history mask, each with shape [4, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features
        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        assert clip_fea is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # TODO: support misaligned inputs 
        x = self.merge_list_of_tensors_to_batch(x)
        ref_latents = self.merge_list_of_tensors_to_batch(ref_latents)
        pose_latents = self.merge_list_of_tensors_to_batch(pose_latents)
        driving_masks = self.merge_list_of_tensors_to_batch(driving_masks)
        ref_masks = self.merge_list_of_tensors_to_batch(ref_masks)

        if history_mask is None:
            x = self.apply_i2v_zeros_masks(x)
        else:
            history_mask = self.merge_list_of_tensors_to_batch(history_mask)
            x = torch.cat([x, history_mask], dim=1)
        ref_latents = self.apply_i2v_ones_masks(ref_latents)
        pose_latents = self.apply_i2v_ones_masks(pose_latents)

        if additional_ref_latents is not None:
            if additional_ref_masks is None:
                raise ValueError("additional_ref_masks is required when additional_ref_latents is provided.")
            additional_ref_latents = self.merge_list_of_tensors_to_batch(additional_ref_latents)
            additional_ref_latents = self.apply_i2v_ones_masks(additional_ref_latents)
            additional_ref_masks = self.merge_list_of_tensors_to_batch(additional_ref_masks)
        elif additional_ref_masks is not None:
            raise ValueError("additional_ref_masks requires additional_ref_latents.")

        B, D, T, H, W = x.shape
        
        assert pose_latents.shape[3] == H//2
        assert pose_latents.shape[4] == W//2
        
        ref_length = 1 * H * W // reduce(mul, self.patch_size)
        seq_length = T * ref_length
        pose_length = T * (H // 2) * (W // 2) // reduce(mul, self.patch_size)

        # embeddings
        x = torch.cat([ref_latents, x], dim=2)
        x = self.patch_embedding(x)
        ref_mask_emb = self.patch_embedding_mask(ref_masks)
        x = x + ref_mask_emb
        pose_emb = self.patch_embedding_pose(pose_latents)
        sam_emb = self.patch_embedding_mask(driving_masks)
        pose_emb = pose_emb + sam_emb
        x = torch.cat(
            [
                rearrange(x, "b c t h w -> b (t h w) c"),
                rearrange(pose_emb, "b c t h w -> b (t h w) c"),
            ],
            dim=1,
        )

        additional_ref_length = 0
        additional_ref_count = 0
        if additional_ref_latents is not None:
            if additional_ref_latents.shape[2] % self.patch_size[0] != 0:
                raise ValueError("additional_ref_latents temporal length must be divisible by temporal patch size.")
            additional_ref_count = additional_ref_latents.shape[2] // self.patch_size[0]
            additional_ref_emb = self.patch_embedding(additional_ref_latents)
            additional_ref_mask_emb = self.patch_embedding_mask(additional_ref_masks)
            additional_ref_emb = additional_ref_emb + additional_ref_mask_emb
            additional_ref_emb = rearrange(additional_ref_emb, "b c t h w -> b (t h w) c")
            additional_ref_length = additional_ref_emb.shape[1]
            x = torch.cat([additional_ref_emb, x], dim=1)

        seq_lens = torch.tensor([u.size(0) for u in x], dtype=torch.long)
        # seq_lens is used for flash attention k_lens

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 (x2) x dim
            context = torch.concat([context_clip, context], dim=1)

        rope_t = T // self.patch_size[0]
        rope_h = H // self.patch_size[1]
        rope_w = W // self.patch_size[2]    

        # grid_sizes:
        # Original spatial-temporal grid dimensions before patching,
        # shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches) 
        grid_sizes = torch.stack([torch.tensor((rope_t, rope_h, rope_w), dtype=torch.long) for _ in range(B)])

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            ref_length=ref_length,
            seq_length=seq_length,
            pose_length=pose_length,
            additional_ref_length=additional_ref_length,
        )
        
        kwargs["rope_T"] = rope_t
        kwargs["rope_H"] = rope_h
        kwargs["rope_W"] = rope_w
        kwargs["hidden_size_head"] = self.hidden_size_head

        kwargs["rope_ref_T"] = {
            "ref": 1,
            "additional_ref": additional_ref_count,
        }

        # TODO: add shift based on rank of sequence parallelism
        base_video_shift = 0 if replace_flag else 1
        kwargs["rope_T_shift"] = {
            "additional_ref": 0,
            "ref": additional_ref_count,
            "pose": base_video_shift + additional_ref_count,
            "video": base_video_shift + additional_ref_count,
        }

        kwargs["rope_H_shift"] = {
            "ref": 120 if replace_flag else 0, 
            "additional_ref": 120 if replace_flag else 0,
            "pose": 0,
            "video": 0,
        }

        kwargs["rope_W_shift"] = {
            "ref": 0,
            "additional_ref": 0, 
            "pose": 120,
            "video": 0,
        }

        def apply_rope_scail(x):
            """
            x: [b, s, n, d]
            """
            y = rope_apply_scail(x, **kwargs)
            return y

        kwargs["rope_apply_func"] = apply_rope_scail

        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes, offset=additional_ref_length + ref_length)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes, offset:int= 0):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            # only keep denoised part of u
            u = u[offset:offset+math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
