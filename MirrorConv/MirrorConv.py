"""
MirrorConv: Parameter-Shared Staggered Topology Operator
(MirrorConv: 參數共享交錯拓撲算子)

GitHub Repository: https://github.com/chihsunchang1225-design/MirrorConv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# Public API Definition 
# (對外公開的 API 定義)
# =====================================================================
__all__ = [
    'Base_MirrorConv',              # Base Core Operator (基礎核心算子 - 供自定義繼承)
    
    # Operators with Dynamic Rectifier (帶動態整流器的模式)
    'MirrorConv_Twin_Rectifier',    # Twin Mode + Rectifier (雙生模式 + 動態整流器)
    'MirrorConv_Chain_Rectifier',   # Chain Mode + Rectifier (鏈式模式 + 動態整流器)
    'MirrorConv_Random_Rectifier',  # Random Mode + Rectifier (隨機模式 + 動態整流器)
    
    # Primary Recommended Operators (純粹模式 - 論文主推 / 次推輕量算子)
    'MirrorConv_Twin',              # Primary Recommended Operator from Paper (論文首推算子 - 雙生模式)
    'MirrorConv_Chain',             # Secondary Recommended Operator from Paper (論文次推算子 - 鏈式模式)
    'MirrorConv_Random',            # General Lightweight Operator (一般輕量化算子 - 隨機模式)
    
    # Internal Sub-Modules (內部構件)
    'MirrorFusion',                 # Feature Fusion and Symmetry Generation (特徵融合與對稱生成)
    'Dynamic_Rectifier',            # Dynamic Rectification (動態整流)
]

# =====================================================================
# Utility Modules 
# (輔助構件)
# =====================================================================
class Rectifier_LeakyReLU(nn.Module):
    """
    Custom LeakyReLU for Dynamic Rectifier bounds.
    (專為動態整流器邊界設計的自定義 LeakyReLU。)
    """
    def __init__(self, low=-2.0, high=9.0, leak=0.1):
        super().__init__()
        self.low = low
        self.high = high
        self.leak = leak

    def forward(self, x):
        # Clamp values within the specified bounds
        # (將數值限制在指定的上下界內)
        clamped = torch.clamp(x, min=self.low, max=self.high)
        # Calculate the residual for the leaky part
        # (計算洩漏部分的殘差)
        residual = x - clamped 
        return clamped + self.leak * residual


# =====================================================================
# Internal Sub-Modules 
# (內部獨立子模組 - 方便提取特徵與加速)
# =====================================================================
class MirrorFusion(nn.Module):
    """
    Block A: Feature Fusion and Symmetry Generation
    (區塊 A：特徵融合與對稱生成)
    """
    def __init__(self, in_channels, half_channels, aligned_out, is_random):
        super().__init__()
        self.is_random = is_random
        self.fusion_gn = nn.GroupNorm(2, aligned_out)
        
        if self.is_random:
            # Initialize two independent weight sets for random mode
            # (為隨機模式初始化兩組獨立的權重)
            self.weight_set_1 = nn.Parameter(torch.Tensor(half_channels, in_channels, 1, 1))
            self.weight_set_2 = nn.Parameter(torch.Tensor(half_channels, in_channels, 1, 1))
            nn.init.kaiming_normal_(self.weight_set_1, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.weight_set_2, mode='fan_out', nonlinearity='relu')
        else:
            # Initialize a single weight set for mirror mode (scaled by 0.707 for variance preservation)
            # (為鏡像模式初始化單一權重，並乘以 0.707 以保持變異數不變)
            self.raw_weights = nn.Parameter(torch.Tensor(half_channels, in_channels, 1, 1))
            nn.init.kaiming_normal_(self.raw_weights, mode='fan_out', nonlinearity='relu')
            with torch.no_grad():
                self.raw_weights.data.mul_(0.707)

    def forward(self, x):
        if self.is_random:
            # Baseline mode: Concatenate two independent randomly initialized weights
            # (Baseline 模式：拼接兩組獨立隨機初始化的權重)
            w_final = torch.cat([self.weight_set_1, self.weight_set_2], dim=0)
        else:
            # Mirror mode: Concatenate weights with their negative values to generate symmetrical features
            # (Mirror 模式：拼接自身與自身的負值，產生對稱特徵)
            w_final = torch.cat([self.raw_weights, -self.raw_weights], dim=0)
            
        fused = F.conv2d(x, w_final)
        return self.fusion_gn(fused)


class Dynamic_Rectifier(nn.Module):
    """
    Block B: Dynamic Rectification
    (區塊 B：動態整流)
    """
    def __init__(self, aligned_out, rectifier_kernel_size, padding):
        super().__init__()
        self.half_channels = aligned_out // 2
        
        # Group 1 (P) -> [scale_P, shift_P], Group 2 (N) -> [scale_N, shift_N]
        # (群組 1 (正向) -> [縮放_P, 偏移_P], 群組 2 (負向) -> [縮放_N, 偏移_N])
        self.Rectifier_conv = nn.Conv2d(
            aligned_out, 4,
            kernel_size=rectifier_kernel_size,
            stride=1,
            padding=padding,
            groups=2, 
            bias=True
        )
        # Initialize weights and biases to 0 for identity mapping at the beginning
        # (將權重與偏置初始化為 0，確保初期為恆等映射)
        with torch.no_grad():
            nn.init.constant_(self.Rectifier_conv.weight, 0)
            nn.init.constant_(self.Rectifier_conv.bias, 0)

        self.Rectifier_gn = nn.GroupNorm(2, 2)
        self.Rectifier_act = Rectifier_LeakyReLU()

    def forward(self, fused):
        Rectifier_features = self.Rectifier_conv(fused)
        
        # Extract scale features and apply activation
        # (提取縮放特徵並套用激活函數)
        raw_scale = Rectifier_features[:, [0, 2], :, :] 
        scale = self.Rectifier_act(self.Rectifier_gn(raw_scale)) + 1.0
        
        # Extract shift features
        # (提取偏移特徵)
        shift = Rectifier_features[:, [1, 3], :, :]
        
        # Expand scale and shift to match channel dimensions
        # (擴展縮放與偏移特徵以匹配通道維度)
        Rectifier_scale = scale.repeat_interleave(self.half_channels, dim=1)
        Rectifier_shift = shift.repeat_interleave(self.half_channels, dim=1)

        # Apply affine transformation
        # (套用仿射變換)
        return (fused * Rectifier_scale) + Rectifier_shift


# =====================================================================
# Main Block 
# (主算子)
# =====================================================================
class Base_MirrorConv(nn.Module):
    """
    Base Mirror Convolution Operator
    Provides the architectural foundation for Twin, Chain, and Random modes.
    (基礎 Mirror 卷積算子：提供 Twin、Chain 與 Random 模式的架構基礎。)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=None, dilation=1, bias=False, 
                 rectifier_kernel_size=None,
                 mode='standard', 
                 twin_mode=True):
        super().__init__()
        
        self.mode = mode 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.twin_mode = twin_mode
        self.dilation = dilation
        
        # Determine configuration based on mode string
        # (根據模式字串決定配置)
        self.is_random = 'full_random' in self.mode
        self.use_Rectifier = 'no_Rectifier' not in self.mode
        
        if rectifier_kernel_size is None:
            rectifier_kernel_size = kernel_size
        if padding is None:
            padding = (kernel_size - 1) * self.dilation // 2

        Rectifier_padding = rectifier_kernel_size // 2
        raw_half = (out_channels + 1) // 2
        
        # Channel parity logic (Ensure correct channel grouping)
        # (通道奇偶數邏輯：確保正確的通道分組)
        if self.twin_mode:
            if raw_half % 2 != 0: raw_half += 1
        else:
            if raw_half % 2 == 0: raw_half += 1
        
        self.half_channels = raw_half
        self.aligned_out = self.half_channels * 2
        
        # ---------------------------------------------------------
        # Block A: Mirror Fusion Layer
        # (區塊 A：鏡像融合層)
        # ---------------------------------------------------------
        self.fusion_block = MirrorFusion(
            in_channels=self.in_channels, 
            half_channels=self.half_channels, 
            aligned_out=self.aligned_out, 
            is_random=self.is_random
        )
        
        # ---------------------------------------------------------
        # Block B: Dynamic Rectifier Layer
        # (區塊 B：動態整流層)
        # ---------------------------------------------------------
        if self.use_Rectifier:
            self.rectifier_block = Dynamic_Rectifier(
                aligned_out=self.aligned_out, 
                rectifier_kernel_size=rectifier_kernel_size, 
                padding=Rectifier_padding
            )
        else:
            # Use nn.Identity as a placeholder to avoid 'if' conditions during forward pass
            # (使用 nn.Identity 作為佔位符，這樣 forward 就不需要寫 if)
            self.rectifier_block = nn.Identity()

        # ---------------------------------------------------------
        # Block C: Topology Decoder Layer (Grouped Spatial Convolution)
        # (區塊 C：拓撲解碼層 - 分組空間卷積)
        # ---------------------------------------------------------
        self.spatial_conv = nn.Conv2d(
            self.aligned_out, self.aligned_out, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=self.dilation,
            groups=self.half_channels, bias=bias 
        )

    # ---------------------------------------------------------
    # Forward pass
    # (向前傳遞)
    # ---------------------------------------------------------
    def forward(self, x):
        fused = self.fusion_block(x)
        fused_adjusted = self.rectifier_block(fused)
        out = self.spatial_conv(fused_adjusted)
        # Slice the output to match the requested output channels exactly
        # (裁切輸出以完全匹配請求的輸出通道數)
        return out[:, :self.out_channels, :, :]


# =====================================================================
# Quick Wrapper Modules 
# (快捷封裝模組)
# =====================================================================

# --- 1. Rectifier Variants (帶動態整流器的變體) ---
class MirrorConv_Twin_Rectifier(Base_MirrorConv):
    """Twin Mode with Dynamic Rectifier (雙生模式 + 動態整流器)"""
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, mode='standard', twin_mode=True, **kwargs)

class MirrorConv_Chain_Rectifier(Base_MirrorConv):
    """Chain Mode with Dynamic Rectifier (鏈式模式 + 動態整流器)"""
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, mode='standard', twin_mode=False, **kwargs)

class MirrorConv_Random_Rectifier(Base_MirrorConv):
    """Random Configuration with Dynamic Rectifier (隨機模式 + 動態整流器)"""
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, mode='full_random', twin_mode=True, **kwargs)

# --- 2. Core Recommended Variants (核心推薦 - 無動態整流器，極致輕量化) ---
class MirrorConv_Twin(Base_MirrorConv):
    """Primary Recommended Operator [Twin Mode] (論文首推算子 - 雙生模式)"""
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, mode='standard_no_Rectifier', twin_mode=True, **kwargs)

class MirrorConv_Chain(Base_MirrorConv):
    """Secondary Recommended Operator [Chain Mode] (論文次推算子 - 鏈式模式)"""
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, mode='standard_no_Rectifier', twin_mode=False, **kwargs)

class MirrorConv_Random(Base_MirrorConv):
    """General Lightweight Operator [Random Mode] (一般輕量化算子 - 隨機模式)"""
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, mode='full_random_no_Rectifier', twin_mode=True, **kwargs)