import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from collections import OrderedDict
import numpy as np
from typing import Optional, Callable

def get_position_embedding_sine(feature_dim=512, num_features=1024, temperature=10000):
    scale = 2 * math.pi
    embed = torch.arange(num_features)
    eps = 1e-6
    embed = embed / (embed[-1:] + eps) * scale
    dim_t = torch.arange(feature_dim, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / feature_dim)

    embed = embed[:,None] / dim_t
    embed = torch.stack((embed[:,0::2].sin(), embed[:,1::2].cos()), dim=2).flatten(1)
    embed = embed.permute(0,1)
    return embed 

class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        drop_attention_rate: float = 0.
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=drop_attention_rate
        )
        self.ln_1 = LayerNorm(d_model)

        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        types: int,
        in_token: int,
        text_token_len: int,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        drop_attention_rate: float = 0.,
        pos_enc: str = 'sine'
    ):
        """
        Transformer module that supports multi-modal inputs including text and auxiliary information.
        
        This transformer is designed to accept multiple types of inputs:
        - Text embeddings
        - Video embeddings  
        - Auxiliary information (e.g., anatomical structure hints)
        
        When some modalities are missing, placeholder tokens are used (typically set to zero)
        for the missing text or auxiliary information inputs.
        """
        super().__init__()
        self.width = width
        self.layers = layers
        self.pos_enc = pos_enc
        self.grad_checkpointing = False

        # Positional embeddings
        if self.pos_enc == 'learned':
            self.pos_embed_video = nn.Parameter(torch.empty(in_token, width))
            nn.init.normal_(self.pos_embed_video, std=0.01)
            self.pos_embed_text = nn.Parameter(torch.empty(text_token_len, width))
            nn.init.normal_(self.pos_embed_text, std=0.01)
        elif self.pos_enc == 'sine':
            pos_embed_video = get_position_embedding_sine(width, text_token_len)
            self.register_buffer('pos_embed_video', pos_embed_video)
            pos_embed_text = get_position_embedding_sine(width, in_token)
            self.register_buffer('pos_embed_text', pos_embed_text)

        # Type embeddings
        self.type_embed = nn.Parameter(torch.empty(types, width))
        nn.init.normal_(self.type_embed, std=0.01)

        # Layer normalization
        self.ln_init = LayerNorm(width)
        self.ln_type_init = LayerNorm(width)
        self.ln_position_init = LayerNorm(width)

        # Transformer blocks
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, act_layer, drop_attention_rate
            ) for _ in range(layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        lang_len: int,
        video_len: int,
        VAalign: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ):
        """
        Forward pass of the Transformer.
        
        Args:
            x: Input tensor of shape [seq_len, batch_size, dim]
            lang_len: Length of text sequence
            video_len: Length of video sequence
            VAalign: Optional alignment for video-text alignment
            attn_mask: Optional attention mask
        """
        # Type embedding
        type_embed_text = self.type_embed[0, None, :].repeat([lang_len, 1, 1])
        type_embed_video = self.type_embed[1, None, :].repeat([video_len, 1, 1])
        type_embed = torch.cat([type_embed_text, type_embed_video], dim=0)[:x.shape[0], :, :]

        # Position embedding
        if VAalign is None:
            pos_embed_text = self.pos_embed_text[:lang_len, None, :]
            pos_embed_video = self.pos_embed_video[:video_len + 1, None, :]
            pos_embed = torch.cat([pos_embed_text, pos_embed_video], dim=0)[:x.shape[0], :, :]
            x = self.ln_init(x) + self.ln_type_init(type_embed) + self.ln_position_init(pos_embed)
        else:
            pos = torch.from_numpy(np.arange(VAalign.shape[-1])).unsqueeze(0).unsqueeze(0).to(VAalign.device)
            midpos = torch.sum(pos * (1. * VAalign), dim=2) / (torch.sum(1. * VAalign, dim=2) + 1e-20)
            midpos = midpos.long().transpose(0, 1)
            pos_embed_text = self.pos_embed_video[midpos.view(-1)].view(midpos.shape[0], midpos.shape[1], -1)
            pos_embed_video = self.pos_embed_video[1:video_len + 1, None, :].repeat(1, midpos.shape[1], 1)
            pos_embed_v = torch.cat([pos_embed_text, pos_embed_video], dim=0)[:x.shape[0], :, :]
            x = self.ln_init(x) + self.ln_type_init(type_embed) + self.ln_position_init(pos_embed_v)

        # Transformer blocks
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = torch.utils.checkpoint.checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

class Keyframe_detect(nn.Module):
    def __init__(self, 
                sim='cos', 
                pos_enc='learned',
                width=256,
                heads=8,
                layers=4,  
                in_token=5*60*4 + 10,
                text_token_len=120,
                types=3,
                fusion_layers=4,
                keyframe_outputdim=2,
                structure_outputdim=5
                ):
        super().__init__()
        self.sim = sim
        self.pos_enc = pos_enc

        # Projections
        self.img_proj_clip = nn.Linear(width, width)
        nn.init.normal_(self.img_proj_clip.weight, std=0.01)

        self.img_proj_text = nn.Linear(width, width)
        nn.init.normal_(self.img_proj_text.weight, std=0.01)

        self.img_proj_ana = nn.Linear(width, width)
        nn.init.normal_(self.img_proj_ana.weight, std=0.01)

        # Fusion module
        self.fusion_module = Transformer(
            width=width,
            layers=fusion_layers,
            heads=heads,
            types=types,
            in_token=in_token,
            text_token_len=text_token_len,
            pos_enc=self.pos_enc
        )

        # self.attn_fusion = MultiHeadAttention(512, p=0.1)

        ### classifier
        self.mlm_projection = nn.Parameter(torch.empty(width, keyframe_outputdim))
        self.mlm_projection_stru = nn.Parameter(torch.empty(width, structure_outputdim))
        nn.init.normal_(self.mlm_projection, std=width ** -0.5)
        nn.init.normal_(self.mlm_projection_stru, std=width ** -0.5)


    def forward(self, video_embed, lang_embed=None, ana_embed=None, VAalign=None):

        ### if multi_modal
        # B, T, D = ana_embed.shape
        # ana_embed = ana_embed + self.img_proj_ana(ana_embed.reshape(B*T, D)).reshape(B, T, -1)
        # B, T, D = lang_embed.shape
        # lang_embed = lang_embed + self.img_proj_text(lang_embed.reshape(B*T, D)).reshape(B, T, -1)
        # B, T, D = video_embed.shape
        # video_embed = video_embed + self.img_proj_clip(video_embed.reshape(B*T, D)).reshape(B, T, -1)

        # x = torch.cat([lang_embed, video_embed], dim=1) 
        # x = self.attn_fusion(x, x, x, VAalign)
        # video_embed = x[:, lang_embed.shape[1]:, :]  

        # x = torch.cat([ana_embed, video_embed], dim=1) 
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.fusion_module(x, ana_embed.shape[1], video_embed.shape[1])
        # x = x.permute(1, 0, 2)  # LND -> NLD
        # video_embed = x[:, ana_embed.shape[1]:, :]  #  
        # out = nn.LogSoftmax(dim=-1)(video_embed @ self.mlm_projection)
        # return out

        B, T, D = video_embed.shape
        video_embed = video_embed + self.img_proj_clip(video_embed.view(B * T, D)).view(B, T, -1)

        # Pad with dummy tokens for missing modalities
        x = torch.cat([torch.zeros_like(video_embed)[:, :2, :], video_embed], dim=1)
        x = x.permute(1, 0, 2)
        x = self.fusion_module(x, 2, video_embed.shape[1])
        x = x.permute(1, 0, 2)
        video_embed = x[:, 2:, :]

        out_key = nn.LogSoftmax(dim=-1)(video_embed @ self.mlm_projection)
        out_stru = nn.LogSoftmax(dim=-1)(video_embed @ self.mlm_projection_stru)
        return out_key, out_stru
