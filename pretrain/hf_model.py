import math
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Callable, Optional
from transformers import CLIPModel, AutoModelForCausalLM, PreTrainedModel, PretrainedConfig, LlamaModel

class VamosConfig(PretrainedConfig):
    model_type = "vamos"
    is_composition = True

    def __init__(self,
                 llm_id: str = 'HuggingFaceH4/zephyr-7b-beta',
                 clip_id: str = 'openai/clip-vit-large-patch14-336',
                 projector_layers: int = 8,
                 projector_heads: int = 16,
                 img_start_token: int = None,
                 **kwargs
                 ):
        super().__init__()
        self.llm_id = llm_id
        self.clip_id = clip_id
        self.projector_layers = projector_layers
        self.projector_heads = projector_heads
        self.img_start_token = img_start_token
        super().__init__(**kwargs)


class VisualAttention(nn.Module):
    """self-attention layer class.
    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, embed_dim, num_heads,
                 bias=True, kdim=None, vdim=None):
        super(VisualAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads

        # Per attention head and per partition values.
        assert embed_dim % num_heads == 0
        self.hidden_size_per_attention_head = embed_dim // num_heads
        self.num_attention_heads_per_partition = num_heads
        self.hidden_size_per_partition = embed_dim

        # Strided linear layer.
        assert self._qkv_same_embed_dim, 'Only Support SelfAttention Currently'
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)

    def forward(self, query, key, value, attn_mask = None):
        # query/key/value: [sq, b, h]
        sq, b, _ = query.size()

        sk = sq
        mixed_x_layer = self.in_proj(query)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + \
            (self.num_attention_heads_per_partition,
             3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        query_layer, key_layer, value_layer = mixed_x_layer.split(
            self.hidden_size_per_attention_head, dim=-1)

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(sq,
            b * self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head).transpose(0, 1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(sk,
            b * self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head).transpose(0, 1)

        q_scaled = query_layer / self.norm_factor
        if attn_mask is not None:
            attention_probs = torch.baddbmm(attn_mask, q_scaled, key_layer.transpose(-2, -1))
        else:
            attention_probs = torch.bmm(q_scaled, key_layer.transpose(-2, -1))
        attention_probs = attention_probs.softmax(dim=-1)

        value_layer = value_layer.view(sk,
            b * self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head).transpose(0, 1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer)

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(b,
            self.num_attention_heads_per_partition,
            sq, self.hidden_size_per_attention_head)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output = self.out_proj(context_layer)

        return output
    
class VisualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.attn = VisualAttention(d_model, n_head)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x, k_x, v_x, attn_mask=attn_mask)

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        x = q_x + self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class Projector(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            out_dim: int,
            mlp_ratio: float = 4.0,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers

        self.resblocks = nn.ModuleList([
            VisualAttentionBlock(
                width, heads, mlp_ratio, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

        self.out = nn.Sequential(
            nn.Linear(width, out_dim),
            nn.LayerNorm(out_dim)
        )

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def get_cast_device(self) -> torch.device:
        return self.resblocks[0].mlp.c_fc.weight.device

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
        return self.out(x)

class VamosPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    is_parallelizable = False
    supports_gradient_checkpointing = True
    config_class = VamosConfig
    base_model_prefix = "transformer"

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        return

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, self.llm.__class__):
            module.gradient_checkpointing = value

class Vamos(PreTrainedModel):
    def __init__(self, config: VamosConfig):
        super().__init__(config)
        self.clip = CLIPModel.from_pretrained(config.clip_id)
        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_id)
        self.projection = Projector(
            self.clip.config.vision_config.hidden_size,
            layers=config.projector_layers,
            heads=config.projector_heads,
            out_dim=self.llm.config.hidden_size
        )
        self.pad_token_id = self.llm.config.pad_token_id

    def get_llm_embeds(self, tokens):
        return self.llm.get_input_embeddings()(tokens)

    def encode_image(self, img):
        """Encodes a batch of images into embeddings.
        Args:
        img: A torch.Tensor of shape (batch_size, num_images, channels, height,
            width).
        Returns:
        A torch.Tensor of shape (batch_size, num_images, 257, embedding_dim).
        """
        bsz, num_imgs, _, _, _ = img.size()
        batch_image_embeds = []
        for i in range(bsz):
            image_embeds = []
            for j in range(num_imgs):
                image_embeds.append(self.projection(self.clip.vision_model(img[i][j].unsqueeze(0))[0]))
            batch_image_embeds.append(torch.cat(image_embeds, dim=0))
        return torch.stack(batch_image_embeds, dim=0)

    def get_prompt_wrap_embeds(self, input_tokens, pos, input_embeds, image_embeds):
        bsz, num_imgs, _, _ = image_embeds.size()
        concat_embeds = []
        for i in range(bsz):
            batch_embed = []
            point = 0
            for j in range(num_imgs):
                batch_embed.append(input_embeds[i][point:pos[i][j]+1])
                batch_embed.append(image_embeds[i][j])
                point = pos[i][j]+1
            batch_embed.append(input_embeds[i][point:])
            concat_embeds.append(torch.cat(batch_embed, dim=0))
        concat_embeds = torch.stack(concat_embeds, dim=0)
        
        concat_targets = []
        tar = input_tokens.masked_fill(input_tokens == self.pad_token_id, -100)
        for i in range(bsz):
            batch_targets = []
            point = 0
            for j in range(num_imgs):
                pre = tar[i][point:pos[i][j]+1]
                img_tar = torch.ones(image_embeds.shape[2],
                            dtype=torch.long).to(image_embeds.device).fill_(-100).reshape(-1)
                batch_targets.append(torch.cat([pre, img_tar], dim=0))
                point = pos[i][j]+1
            batch_targets.append(tar[i][point:])
            concat_targets.append(torch.cat(batch_targets, dim=0))
        concat_targets = torch.stack(concat_targets, dim=0)
        return concat_embeds, concat_targets

    def forward(self,
                input_ids: torch.tensor,
                image: torch.tensor = None,
                ):
        """
        Args:
            input_ids: torch.tensor(bsz, seq_len)
            image: torch.tensor(bsz, num_imgs, 3, 224, 224)
        """
        if image is not None:
            image_embeds = self.encode_image(image)
            bsz, num_imgs, _, _ = image_embeds.size()
            img_start_pos = torch.where(input_ids == self.config.img_start_token)[-1].reshape(-1, num_imgs)

        input_embeds = self.get_llm_embeds(input_ids)
        if image is not None:
            input_embeds, targets = self.get_prompt_wrap_embeds(input_ids, img_start_pos, input_embeds, image_embeds)
        if image is None:
            targets = input_ids.masked_fill(input_ids == self.pad_token_id, -100)
        attention_mask = torch.ones(targets.shape, dtype=torch.long).to(targets.device)
        
        outputs = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=targets,
            return_dict=True
        )
        return outputs

    def generate(self,
                input_ids: torch.tensor,
                image: torch.tensor = None,
                **generate_kwargs
                ):
        """
        Args:
            input_ids: torch.tensor(bsz, seq_len)
            image: torch.tensor(bsz, num_imgs, 3, 224, 224)
        """
        if image is not None:
            image_embeds = self.encode_image(image)
            bsz, num_imgs, _, _ = image_embeds.size()
            img_start_pos = torch.where(input_ids == self.config.img_start_token)[-1].reshape(bsz, num_imgs)
            
        input_embeds = self.get_llm_embeds(input_ids)
        if image is not None:
            input_embeds, _ = self.get_prompt_wrap_embeds(input_ids, img_start_pos, input_embeds, image_embeds)
        attention_mask = torch.ones(input_embeds.shape[:-1], dtype=torch.long).to(input_embeds.device)
    
        outputs = self.llm.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            **generate_kwargs
        )
        return outputs
    