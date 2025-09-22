import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class D3PM(nn.Module):
    def __init__(self): # Add any required parameters
        super().__init__()
        # Define your model architecture here

class ConditionalD3PM(nn.Module):
    def __init__(self, num_classes): # Add any required parameters
        super().__init__()
        self.num_classes = num_classes
        # Define your conditional model architecture here

# class taken from -> https://github.com/lucidrains/denoising-diffusion-pytorch/issues/249
class SinusoidalPosEmb(nn.Module):
    def __init__(self, *, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class CEN(nn.Module):
    def __init__(self, *, num_classes, hidden_dim=128, out_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.dense_layer = nn.Conv2d(2*out_dim, out_dim, kernel_size=1)
    
    def forward(self, y, h):
        x = self.embedding(y)
        x = self.mlp(x)
        x = x[:, :, None, None].expand(-1, -1, h.shape[-2], h.shape[-1])
        h = self.dense_layer(torch.cat([x, h], dim=1))
        return h

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channel, out_channel=None, dropout=0, conv_short=False, base_channel_size=128):
        super().__init__()
        self.out_channel = out_channel or in_channel
        self.dropout = dropout
        self.in_channel = in_channel
        self.base_channel_size = base_channel_size

        # First norm + convolution...
        self.norm1 = nn.GroupNorm(32, self.in_channel)
        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, padding=1)

        # projecting temb layer...
        self.temb_proj = nn.Linear(self.base_channel_size*4, self.out_channel)

        # Second norm + convolution...
        self.norm2 = nn.GroupNorm(32, self.out_channel)
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, padding=1)

        # skip connection...
        if in_channel != out_channel:
            if conv_short:
                self.skip_connection = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, padding=1)
            else:
                self.skip_connection = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()
        
    def forward(self, x, temb):
        # Norm + SiLU...
        h = F.silu(self.norm1(x))
        # first convolution...
        h = self.conv1(h)
        # adding timestep embedding...
        temb = self.temb_proj(F.silu(temb))[:, :, None, None] # [B, out, 1, 1]
        # encorporating timestemp embedding with the input...
        h = h + temb
        # Norm + SiLU...
        h = F.silu(self.norm2(h))
        # Dropout...
        h = F.dropout(h, p=self.dropout, training=self.training)
        # second conv
        h = self.conv2(h)
        # residual addition
        x = self.skip_connection(x)
        h = x+h
        return h

class Attention_Block(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channel)
        self.q = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.k = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.v = nn.Conv2d(in_channel, in_channel, kernel_size=1)

        self.proj_out = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
    
    def forward(self, x):
        B, C, H, W = x.shape

        h = self.norm(x)

        q = self.q(h)
        q_flat = q.view(B, C, H*W)
        k = self.k(h)
        k_flat = k.view(B, C, H*W)
        v = self.v(h)
        v_flat = v.view(B, C, H*W)

        w = torch.einsum('bci,bcj->bij', q_flat, k_flat) * (int(C) ** (-0.5))
        w = F.softmax(w, dim=-1) 
        
        h = torch.einsum('bij,bcj->bci', w, v_flat)
        h = h.view(B, C, H, W)
        h = self.proj_out(h)

        return x+h

class Downsampling_Block(nn.Module):
    def __init__(self, *, in_channel, base_channel_size, ch_mult, num_resnet_blocks, attn_resolutions, dropout, downsample_with_conv=False, conditional=False, num_classes=None):
        super().__init__()
        self.num_levels = len(ch_mult)
        self.num_resnet_blocks = num_resnet_blocks
        self.conditional = conditional
        self.num_classes = num_classes
        
        self.conv_first = nn.Conv2d(in_channel, base_channel_size, kernel_size=3, padding=1)

        self.res_blocks = nn.ModuleList([
            nn.ModuleList() for _ in range(self.num_levels)
        ])

        self.attn_resolutions = attn_resolutions
        self.attn_blocks = nn.ModuleList([
            nn.ModuleList() for _ in range(self.num_levels)
        ])

        if self.conditional:
            self.class_emb_blocks = nn.ModuleList([
                nn.ModuleList() for _ in range(self.num_levels)
            ])

        for i_level in range(self.num_levels):
            for i_block in range(self.num_resnet_blocks):
                if i_block:
                    temp_in_channel = base_channel_size*ch_mult[i_level]
                else:
                    temp_in_channel = base_channel_size*ch_mult[i_level-1] if i_level > 0 else base_channel_size
                
                self.res_blocks[i_level].append(ResnetBlock(in_channel=temp_in_channel, 
                                                            out_channel=base_channel_size*ch_mult[i_level], 
                                                            dropout=dropout,
                                                            base_channel_size=base_channel_size))
                self.attn_blocks[i_level].append(Attention_Block(in_channel=base_channel_size*ch_mult[i_level]
                                                                ))
                if self.conditional:
                    self.class_emb_blocks[i_level].append(CEN(num_classes=self.num_classes,
                                                     out_dim=base_channel_size*ch_mult[i_level]))

        if downsample_with_conv:
            self.down = nn.Conv2d(base_channel_size*ch_mult[-1], base_channel_size*ch_mult[-1], kernel_size=3, stride=2)
        else:
            self.down = nn.AvgPool2d(2)
        
    def forward(self, x, temb, class_type=None):
        h = self.conv_first(x)
        hs = [h]

        for i_level in range(self.num_levels):
            for i_block in range(self.num_resnet_blocks):
                h = self.res_blocks[i_level][i_block](hs[-1], temb)
                if h.shape[2] in self.attn_resolutions:
                    h = self.attn_blocks[i_level][i_block](h)
                if self.conditional:
                   h = self.class_emb_blocks[i_level][i_block](class_type, h)
                hs.append(h)
            if i_level != self.num_levels - 1:
                hs.append(self.down(hs[-1]))

        return hs[-1], hs

class Upsampling_Block(nn.Module):
    def __init__(self, *, in_channel, base_channel_size, ch_mult, num_resnet_blocks, attn_resolutions, dropout, upsample_with_conv=False, conditional=False, num_classes=None):
        super().__init__()
        self.num_levels = len(ch_mult)
        self.num_resnet_blocks = num_resnet_blocks
        self.conditional = conditional
        self.num_classes = num_classes
        
        self.res_blocks = nn.ModuleList([
            nn.ModuleList() for _ in range(self.num_levels)
        ])

        self.attn_resolutions = attn_resolutions
        self.attn_blocks = nn.ModuleList([
            nn.ModuleList() for _ in range(self.num_levels)
        ])

        if self.conditional:
            self.class_emb_blocks = nn.ModuleList([
                nn.ModuleList() for _ in range(self.num_levels)
            ])

        for i_level in reversed(range(self.num_levels)):
            for i_block in range(self.num_resnet_blocks+1):
                if i_block == self.num_resnet_blocks:
                    temp_in_channel = base_channel_size*(ch_mult[i_level-1]+ch_mult[i_level]) if i_level > 0 else 2*base_channel_size
                elif (i_level != self.num_levels-1 and i_block == 0):
                    temp_in_channel = base_channel_size*(ch_mult[i_level]+ch_mult[i_level+1])
                else:
                    temp_in_channel = 2*base_channel_size*ch_mult[i_level]
                
                self.res_blocks[i_level].append(ResnetBlock(in_channel=temp_in_channel, 
                                                            out_channel=base_channel_size*ch_mult[i_level], 
                                                            dropout=dropout,
                                                            base_channel_size=base_channel_size))
                self.attn_blocks[i_level].append(Attention_Block(in_channel=base_channel_size*ch_mult[i_level]))
                if self.conditional:
                    self.class_emb_blocks[i_level].append(CEN(num_classes=self.num_classes,
                                                          out_dim=base_channel_size*ch_mult[i_level]
                                                          ))
        if upsample_with_conv:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channel, in_channel, kernel_size = 3, padding=1)
            )
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, hs, temb, class_type=None):
        h = x
        for i_level in reversed(range(self.num_levels)):
            for i_block in range(self.num_resnet_blocks+1):
                # for managing the mismatch...
                if h.shape[-2: ] != hs[-1].shape[-2: ]:
                   diff_h = hs[-1].shape[2] - h.shape[2]
                   diff_w = hs[-1].shape[3] - h.shape[3]
                   h = F.pad(h, (0, diff_w, 0, diff_h))
                h = self.res_blocks[i_level][i_block](torch.cat([h, hs.pop()], dim = 1), temb)
                if h.shape[2] in self.attn_resolutions:
                    h = self.attn_blocks[i_level][i_block](h)
                if self.conditional:
                    h = self.class_emb_blocks[i_level][i_block](class_type, h)
            if i_level != 0:
                h = self.upsample(h)

        return h

# the default value for these is taken from the scripts of the diffusion paper...
class DDPM(nn.Module):
    def __init__(self, *, in_channel, out_channel, base_channel_size=128, ch_mult=[1, 2, 4, 8], num_resnet_blocks=2, attn_resolutions=(14, 7, 3, 1,), dropout=0): # Add any required parameters
        super().__init__()
        # Define your model architecture here
        self.base_channel_size = base_channel_size
        self.dropout = dropout
        self.ch_mult = ch_mult
        self.num_resnet_blocks = num_resnet_blocks
        self.in_channel = in_channel
        self.attn_resolutions = attn_resolutions
        self.out_channel = out_channel
        
        # downsample block...
        self.downsampler = Downsampling_Block(in_channel=self.in_channel,
                                              base_channel_size=self.base_channel_size,
                                              ch_mult=self.ch_mult,
                                              num_resnet_blocks=self.num_resnet_blocks,
                                              attn_resolutions=self.attn_resolutions,
                                              dropout=self.dropout
                                              )

        # bottleneck layers...
        self.resnet_block_mid1 = ResnetBlock(in_channel=self.base_channel_size*ch_mult[-1],
                                             dropout=dropout)
        self.resnet_block_mid2 = ResnetBlock(in_channel=self.base_channel_size*ch_mult[-1],
                                             dropout=dropout)
        self.attn_block = Attention_Block(in_channel=self.base_channel_size*ch_mult[-1])

        # upsample block along with skip connections from the decoder block...
        self.upsampler = Upsampling_Block(in_channel=self.in_channel,
                                          base_channel_size=self.base_channel_size,
                                          ch_mult=self.ch_mult,
                                          num_resnet_blocks=self.num_resnet_blocks,
                                          attn_resolutions=self.attn_resolutions,
                                          dropout=self.dropout
                                         )

        # time step embedding...
        self.temb = nn.Sequential(
            SinusoidalPosEmb(dim=self.base_channel_size),
            nn.Linear(self.base_channel_size, self.base_channel_size*4),
            nn.SiLU(),
            nn.Linear(self.base_channel_size*4, self.base_channel_size*4) # [B, base_channel_size*4]
        )

        self.final_norm = nn.GroupNorm(32, self.base_channel_size)
        self.final_conv = nn.Conv2d(self.base_channel_size, self.out_channel, kernel_size=3, padding=1)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)
    
    def forward(self, x, t):
        time_emb = self.temb(t)
        x, hs = self.downsampler(x, time_emb)
        x = self.resnet_block_mid1(x, time_emb)
        x = self.attn_block(x)
        x = self.resnet_block_mid2(x, time_emb)
        x = self.upsampler(x, hs, time_emb)
        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)
        return x

class ConditionalDDPM(nn.Module):
    def __init__(self, *, in_channel, out_channel, num_classes, base_channel_size=128, ch_mult=[1, 2, 4, 8], num_resnet_blocks=2, attn_resolutions=(14, 7, 3, 1,), dropout=0): # Add any required parameters
        super().__init__()
        self.num_classes = num_classes
        # Define your conditional model architecture here
        self.base_channel_size = base_channel_size
        self.dropout = dropout
        self.ch_mult = ch_mult
        self.num_resnet_blocks = num_resnet_blocks
        self.in_channel = in_channel
        self.attn_resolutions = attn_resolutions
        self.out_channel = out_channel
        
        # downsample block...
        self.downsampler = Downsampling_Block(in_channel=self.in_channel,
                                              base_channel_size=self.base_channel_size,
                                              ch_mult=self.ch_mult,
                                              num_resnet_blocks=self.num_resnet_blocks,
                                              attn_resolutions=self.attn_resolutions,
                                              dropout=self.dropout,
                                              conditional=True,
                                              num_classes=self.num_classes
                                              )

        # bottleneck layers...
        self.resnet_block_mid1 = ResnetBlock(in_channel=self.base_channel_size*ch_mult[-1],
                                             dropout=dropout)
        self.resnet_block_mid2 = ResnetBlock(in_channel=self.base_channel_size*ch_mult[-1],
                                             dropout=dropout)
        self.attn_block = Attention_Block(in_channel=self.base_channel_size*ch_mult[-1])

        # upsample block along with skip connections from the decoder block...
        self.upsampler = Upsampling_Block(in_channel=self.in_channel,
                                          base_channel_size=self.base_channel_size,
                                          ch_mult=self.ch_mult,
                                          num_resnet_blocks=self.num_resnet_blocks,
                                          attn_resolutions=self.attn_resolutions,
                                          dropout=self.dropout,
                                          conditional=True,
                                          num_classes=self.num_classes
                                         )

        # time step embedding...
        self.temb = nn.Sequential(
            SinusoidalPosEmb(dim=self.base_channel_size),
            nn.Linear(self.base_channel_size, self.base_channel_size*4),
            nn.SiLU(),
            nn.Linear(self.base_channel_size*4, self.base_channel_size*4) # [B, base_channel_size*4]
        )

        self.final_norm = nn.GroupNorm(32, self.base_channel_size)
        self.final_conv = nn.Conv2d(self.base_channel_size, self.out_channel, kernel_size=3, padding=1)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x, t, class_type):
        time_emb = self.temb(t)
        x, hs = self.downsampler(x, time_emb, class_type)
        x = self.resnet_block_mid1(x, time_emb)
        x = self.attn_block(x)
        x = self.resnet_block_mid2(x, time_emb)
        x = self.upsampler(x, hs, time_emb, class_type)
        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)
        return x
    
        