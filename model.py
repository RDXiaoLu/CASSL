from base import *
from models.vgg16 import VGG16
from collections import OrderedDict
from timm.models.layers import drop_path, trunc_normal_
from untils import tokenize
from torch.nn.utils import spectral_norm
import torch
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.spacial_dim = spacial_dim

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        cls_pos = self.positional_embedding[0:1, :]
        # spatial_pos = F.interpolate(self.positional_embedding[1:,].reshape(1, self.spacial_dim, self.spacial_dim, self.embed_dim).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = self.positional_embedding[1:].reshape(self.spacial_dim, self.spacial_dim, self.embed_dim)[:H, :W]
        spatial_pos = spatial_pos.reshape(-1, self.embed_dim)
        # spatial_pos = spatial_pos.reshape(self.embed_dim, H*W).permute(1, 0)
        positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)

        x = x + positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        x = x.permute(1, 2, 0)
        global_feat = x[:, :, 0]
        feature_map = x[:, :, 1:].reshape(B, -1, H, W)
        return global_feat, feature_map

class CLIPResNetWithAttention(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim=1024, input_resolution=224, width=64, pretrained="/data/yuanhui/wwbl_dense_two/RN50.pt", att_level3=False, baseline=False, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        # self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, 32, output_dim)
        self.att_level3 = att_level3
        self.baseline = baseline

        self.reduce_channels = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(512)
        

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

                    if 'positional_embedding' in new_k:
                        if self.attnpool.positional_embedding.shape != state_dict[new_k].shape:
                            print(f'Resize the pos_embed shape from {state_dict[new_k].shape} to {self.attnpool.positional_embedding.shape}')
                            cls_pos = state_dict[new_k][0:1, :]
                            H = W = self.input_resolution // 32
                            spatial_pos = F.interpolate(state_dict[new_k][1:,].reshape(1, 7, 7, cls_pos.shape[1]).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
                            spatial_pos = spatial_pos.reshape(cls_pos.shape[1], H*W).permute(1, 0)
                            positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                            state_dict[new_k] = positional_embedding
                            assert self.attnpool.positional_embedding.shape == state_dict[new_k].shape

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in CLIPResNet')

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)

        outs = []
        x = self.layer1(x)
        outs.append(x)
        x = self.layer2(x)
        outs.append(x)
        x = self.layer3(x)
        outs.append(x)
        x = self.layer4(x)
        outs.append(x)

        x_global, x_local = self.attnpool(x)
        outs.append([x_global, x_local])
        if self.att_level3:
            new_outs = [outs[0], outs[1], outs[2], outs[4][1], outs[4]]
            if self.baseline:
                new_outs = new_outs[:-1]
            return tuple(new_outs)
        else:
            return tuple(outs)
        

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
    ):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        x = x + self.cross_attn(q, mem, mem)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x

class ContextDecoder(nn.Module):
    def __init__(self,
                 transformer_width=256,
                 transformer_heads=4,
                 transformer_layers=6,
                 visual_dim=1024,
                 dropout=0.1,
                 **kwargs):
        super().__init__()

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
        )

        self.decoder = nn.ModuleList([
                    TransformerDecoderLayer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)
                ])
        
        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    
    def forward(self, text, visual):
        B, N, C = visual.shape
        visual = self.memory_proj(visual)
        x = self.text_proj(text)

        for layer in self.decoder:
            x = layer(x, visual)
        
        return self.out_proj(x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_path=0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, drop_path_rate=0.):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, dpr[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class CLIPTextContextEncoder(nn.Module):
    def __init__(self, context_length=40,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=1024,
                 out_dim=256,
                 pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.embed_dim = embed_dim

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                    state_dict[k] = checkpoint[k]
                
                if k == 'positional_embedding' or k == 'text_projection' or k.startswith('token_embedding') or k.startswith('ln_final'):
                    if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                        checkpoint[k] = checkpoint[k][:self.context_length]
                        print('positional_embedding is tuncated from 77 to', self.context_length)
                    state_dict[k] = checkpoint[k]
             
            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in text encoder')


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text, context):
        x_text = self.token_embedding(text)  # n_clas, n_text, C
        K, N1, C = x_text.shape
        B, N2, C = context.shape

        eos_indx = text.argmax(dim=-1) + N2
        eos_indx = eos_indx.reshape(1, K).expand(B, K).reshape(-1)

        x_text = x_text.reshape(1, K, N1, C).expand(B, K, N1, C)
        context = context.reshape(B, 1, N2, C).expand(B, K, N2, C)

        x = torch.cat([x_text[:,:,0:1], context, x_text[:, :, 1:]], dim=2).reshape(B*K, N1+N2, C)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("x contains NaN or Inf values")
        x = x + self.positional_embedding
        if torch.isnan(self.positional_embedding).any() or torch.isinf(self.positional_embedding).any():
            print("self.positional_embedding contains NaN or Inf values")
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
        x = x.reshape(B, K, self.embed_dim)
        return x
def snlinear(in_features, out_features):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- utils : Self-Support Prototype (soft, vectorized) ---------------- #
class PrototypeRefiner(nn.Module):
 
    def __init__(self, temp=10., use_soft=True, eps=1e-5):
        super().__init__()
        self.temp = temp
        self.use_soft = use_soft
        self.eps = eps

      
        self.fg_m = nn.Parameter(torch.tensor(0.5)) 
        self.bg_m = nn.Parameter(torch.tensor(0.3))  

    def forward(self, feat, logits, fg_proto=None, bg_proto=None):
        B, C, H, W = feat.shape
        prob = logits.softmax(dim=1)              
        w_fg = prob[:, 1:2]                         
        w_bg = prob[:, 0:2][:, 0:1]              

     
        feat_flat = feat.view(B, C, -1)              
        w_fg_flat = w_fg.view(B, 1, -1)              
        w_bg_flat = w_bg.view(B, 1, -1)              

        num_fg = torch.einsum('bcn,bin->bc', feat_flat, w_fg_flat)  
        den_fg = w_fg_flat.sum(dim=-1) + 1e-5                   
        num_bg = torch.einsum('bcn,bin->bc', feat_flat, w_bg_flat)  
        den_bg = w_bg_flat.sum(dim=-1) + 1e-5                      

        new_fg = (num_fg / den_fg).view(B, C, 1, 1)               
        new_bg = (num_bg / den_bg).view(B, C, 1, 1)                


        if fg_proto is None:
            fg_proto = new_fg
            bg_proto = new_bg
        else:
            fg_m = torch.sigmoid(self.fg_m)
            bg_m = torch.sigmoid(self.bg_m)
            fg_proto = fg_m * fg_proto + (1 - fg_m) * new_fg
            bg_proto = bg_m * bg_proto + (1 - bg_m) * new_bg

        sim_fg = F.cosine_similarity(feat, fg_proto, dim=1)  # B×H×W
        sim_bg = F.cosine_similarity(feat, bg_proto, dim=1)  # B×H×W
        logits_ref = torch.stack([sim_bg, sim_fg], dim=1) * self.temp

        return logits_ref, fg_proto.detach(), bg_proto.detach()


class MultiModel(nn.Module):
    def __init__(self,
                 args,
                 
                 text_dim=1024,
                 context_length=40,
                 token_embed_dim=512,
                 class_names=None,
                 tau=0.07):
        super().__init__()
        self.class_names = class_names or []
        self.num_classes = len(self.class_names)
        self.tau = tau

     
        self.E = VGG16()
        self.D = MMDecoder([2048 + 1], 1,
                           out_size=(int(args["Isize"]), int(args["Isize"])),
                           is_blip=False)

        self.backbone = CLIPResNetWithAttention(layers=[3, 4, 6, 3],
                                                output_dim=1024,
                                                input_resolution=1344,
                                                pretrained='.yourpath/RN50.pt')
        self.backbone.init_weights()


        self.text_encoder = CLIPTextContextEncoder(context_length=40,
                                                   embed_dim=1024,
                                                   transformer_width=512,
                                                   transformer_heads=8,
                                                   transformer_layers=12,
                                                   pretrained='.yourpath/RN50.pt')
        self.text_encoder.init_weights()

        self.context_decoder = ContextDecoder(transformer_width=256,
                                              transformer_heads=4,
                                              transformer_layers=3,
                                              visual_dim=1024,
                                              dropout=0.1)

   
        ctx_len = self.text_encoder.context_length - context_length
        self.contexts = nn.Parameter(torch.randn(1, ctx_len, token_embed_dim))
        nn.init.trunc_normal_(self.contexts)

        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)
        self.texts = torch.cat([tokenize(c, context_length=context_length) for c in self.class_names])

   
        self.attn = nn.MultiheadAttention(embed_dim=1024, num_heads=4) 
        self.t_ln = nn.LayerNorm(1024)
        self.alpha = nn.Parameter(torch.tensor(0.5))  

       
        self.proto_refiner = PrototypeRefiner(temp=10., use_soft=True)

     
        self.z_proj = nn.Linear(512, 1024)


    def compute_text_features(self, x, dummy=False):
        global_feat, visual_embed = x[4]
        B, C, H, W = visual_embed.shape
        visual_ctx = torch.cat([global_feat[:, :, None],
                                visual_embed.flatten(2)], dim=2).permute(0, 2, 1)  # B×N×C

        if dummy:
            txt = torch.randn(B, self.num_classes, C, device=global_feat.device)
        else:
            txt = self.text_encoder(self.texts.to(global_feat.device),
                                    self.contexts).expand(B, -1, -1)  

        diff = self.context_decoder(txt, visual_ctx)                
        T_base = txt + self.gamma * diff                            
        return T_base, txt                                        

    def compute_score_maps(self, x, text_feat):
        _, visual_embeddings = x[4]                                  
        t = F.normalize(text_feat, dim=-1)
        v = F.normalize(visual_embeddings, dim=1)
        score = torch.einsum('bchw,bkc->bkhw', v, t) / self.tau    
        score_0 = F.interpolate(score, x[0].shape[2:], mode='bilinear', align_corners=False)
        return score, score_0

    # ------------------------------------------------------ #
    def forward(self, image, z_text, dummy=False):
        x = self.backbone(image)                                      
        global_feat, visual_embed = x[4]                           

   
        T_base, T_ori = self.compute_text_features(x, dummy=dummy)   
        dtype = self.z_proj.weight.dtype
        device = self.z_proj.weight.device
        z_text_1024 = self.z_proj(z_text.to(device=device, dtype=dtype)).unsqueeze(1)
      
        top_prompt_merge = T_base * global_feat.unsqueeze(1)         
        ori_prompt_merge = z_text_1024 * T_ori                       
        sign_merge = torch.sgn(ori_prompt_merge)

       
        attn_in  = sign_merge.permute(1, 0, 2).contiguous()          
        attn_ctx = top_prompt_merge.permute(1, 0, 2).contiguous()     
        attn_out, _ = self.attn(attn_in, attn_ctx, attn_ctx)          
        attn_out = attn_out.permute(1, 0, 2).contiguous()            

        
        T_fused = torch.sigmoid(self.alpha) * T_base + (1 - torch.sigmoid(self.alpha)) * self.t_ln(attn_out)

        
        score_3, score_0 = self.compute_score_maps(x, T_fused)       

        
        z_n = F.normalize(z_text_1024.squeeze(1), dim=-1)            
        T_n = F.normalize(T_fused, dim=-1)                           
        # B×K = (B×1024) · (B×1024×K)
        cls_sim = torch.einsum('bc,bkc->bk', z_n, T_n)              
        tgt_idx = cls_sim.argmax(dim=1)                              

       
        B, K, h, w = score_3.shape
        gather_idx = tgt_idx.view(B, 1, 1, 1).expand(B, 1, h, w)     
        sel_score = score_3.gather(dim=1, index=gather_idx)          
        
        logits0 = torch.cat([1 - sel_score.sigmoid(), sel_score.sigmoid()], dim=1) 

        
        logits1, fg_p, bg_p = self.proto_refiner(visual_embed, logits0)
        logits2, _, _       = self.proto_refiner(visual_embed, logits1, fg_p, bg_p)
        logits_final = 0.7 * logits2 + 0.3 * logits1

       
        h3, w3 = x[3].shape[2:]
        logits_up = F.interpolate(
        logits_final[:, 1:2], size=(h3, w3),
        mode='bilinear', align_corners=False
        )
        x_list = list(x[:-1])
        x_list[3] = torch.cat([x_list[3], logits_up], dim=1)

       
        mask = self.D(x_list[3], z_text)
        return mask

 
    @staticmethod
    def consistency_kl(score_new, score_old):
        """
        KL(softmax(new) || softmax(old)) over class dim.
        score_*: B×K×H×W
        """
        p = F.log_softmax(score_new, dim=1)
        q = F.softmax(score_old, dim=1)
        return F.kl_div(p, q, reduction='batchmean')

