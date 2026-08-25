# MirrorConv: Parameter-Shared Staggered Topology Operator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

This is the official PyTorch implementation of **MirrorConv**, a lightweight convolutional operator designed for Extreme Lightweight Vision Models. 

### Installation

For the fastest setup, you can install MirrorConv directly from GitHub using `pip`. Choose one of the methods below:

**Method 1: Using git (Recommended)**
```bash
pip install git+https://github.com/chihsunchang1225-design/MirrorConv.git
```

**Method 2: Direct download (If you don't have git installed)**
```bash
pip install https://github.com/chihsunchang1225-design/MirrorConv/archive/refs/heads/main.zip
```

*(Note: We recommend installing it within a virtual environment.)*

## 💻 Quick Start

MirrorConv provides several variants of the Parameter-Shared Staggered Topology Operator. You can easily integrate them into any existing PyTorch model as a drop-in replacement for standard spatial convolutions.

### Basic Usage (Primary Recommended)
The paper primarily recommends **`MirrorConv_Twin`** for extreme lightweight vision models. The API design closely follows `torch.nn.Conv2d`.

```python
import torch
from MirrorConv import MirrorConv_Twin

# 1. Initialize the operator
layer = MirrorConv_Twin(
    in_channels=32, 
    out_channels=64, 
    kernel_size=3, 
)

# 2. Create dummy input tensor (Batch, Channels, Height, Width)
x = torch.randn(1, 32, 224, 224)

# 3. Forward pass
output = layer(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")
# Expected Output: torch.Size([1, 64, 224, 224])
```

---

## 📚 Detailed API Reference & Usage Guide

Depending on your experimental needs, MirrorConv exposes multiple modules ranging from highly optimized wrappers to independent building blocks.

### 1. Primary Recommended Variants (No Dynamic Rectifier)
**🟢 Highly recommended for edge devices.** These are the most efficient variants and the final recommended configurations based on our empirical studies.

*   `MirrorConv_Twin`: Primary recommended operator (Twin mode).
*   `MirrorConv_Chain`: Secondary recommended operator (Chain mode).
*   `MirrorConv_Random`: General lightweight baseline (Random mode).

**Usage Example:**
```python
from MirrorConv import MirrorConv_Twin

# The API signature is identical for Twin, Chain, and Random modes (No Rectifier)
layer = MirrorConv_Twin(
    in_channels=32,             # [Required] Number of input channels (int)
    out_channels=64,            # [Required] Number of output channels (int)
    kernel_size=3,              # [Optional] Default: 3
    stride=1,                   # [Optional] Default: 1
    padding=None,               # [Optional] Default: None (Auto-calculated to preserve spatial dimensions)
    dilation=1,                 # [Optional] Default: 1
    bias=False                  # [Optional] Default: False
)
```

### 2. Exploratory & Ablation Variants (With Dynamic Rectifier)
**🟠 For researchers interested in boundary exploration.** These variants include the Dynamic Rectification block. Extensive experiments demonstrate they do not always provide the optimal efficiency-accuracy trade-off, but they are useful for ablation studies.

*   `MirrorConv_Twin_Rectifier`
*   `MirrorConv_Chain_Rectifier`
*   `MirrorConv_Random_Rectifier`

**Usage Example:**
```python
from MirrorConv import MirrorConv_Twin_Rectifier

# These variants include the additional 'rectifier_kernel_size' parameter
layer_with_rectifier = MirrorConv_Twin_Rectifier(
    in_channels=32,             # [Required] Number of input channels
    out_channels=64,            # [Required] Number of output channels
    kernel_size=3,              # [Optional] Main spatial kernel size. Default: 3
    stride=1,                   # [Optional] Default: 1
    padding=None,               # [Optional] Default: None
    dilation=1,                 # [Optional] Default: 1
    bias=False,                 # [Optional] Default: False
    rectifier_kernel_size=3     # [Optional] Kernel size for the dynamic rectifier. 
                                # If None, defaults to kernel_size.
)
```

### 3. Core Base Operator
If you want complete parametric control over the architectural behavior, you can call the base class directly.

```python
from MirrorConv import Base_MirrorConv

base_layer = Base_MirrorConv(
    in_channels=32,
    out_channels=64,
    kernel_size=3,
    
    # --- Base_MirrorConv Exclusive Architectural Controls ---
    mode='standard',            # Mode string. Default: 'standard'
                                # Options: 'standard', 'full_random', 'standard_no_Rectifier', 'full_random_no_Rectifier'
    twin_mode=True              # Channel parity logic. Default: True
                                # True: Twin mode logic / False: Chain mode logic
)
```

### 4. Internal Sub-Modules (For Customization)
To facilitate deep ablation studies and custom architectural designs, we explicitly expose the underlying building blocks.

**Block A: MirrorFusion (Feature Fusion and Symmetry Generation)**
```python
from MirrorConv import MirrorFusion

fusion_block = MirrorFusion(
    in_channels=32,             # [Required]
    half_channels=16,           # [Required] Usually half of out_channels
    aligned_out=32,             # [Required] Usually half_channels * 2
    is_random=False             # [Required] False (Mirror mode) or True (Random mode)
)
fused_features = fusion_block(torch.randn(1, 32, 224, 224))
```

**Block B: Dynamic_Rectifier (Dynamic Rectification)**
```python
from MirrorConv import Dynamic_Rectifier

rectifier_block = Dynamic_Rectifier(
    aligned_out=32,             # [Required] Channels aligned from the Fusion block
    rectifier_kernel_size=3,    # [Required] Spatial kernel size for the rectifier
    padding=1                   # [Required] Usually rectifier_kernel_size // 2
)
# Recommended input: features already processed by Fusion.
adjusted_features = rectifier_block(fused_features)
```

## 📖 Citation

If you find MirrorConv useful in your research, we kindly request that you cite our primary journal paper (currently under review). If you are specifically referencing our early findings or presentation, please cite the ARIS 2026 conference paper.

**[Primary Citation] Journal Paper (Under Review):**
> The core architecture and comprehensive experiments have been submitted to a journal. Citation details will be updated here upon official acceptance.

**[Secondary] Conference Paper (ARIS 2026):**
```bibtex
@inproceedings{2026mirrorconv_conf,
  title={MirrorConv: A Parameter-Shared Staggered Topology Operator for Extreme Lightweight Vision Models and its Empirical Robustness Analysis},
  author={Shih, Jia-Shing and Chang, Chi-Hsun},
  booktitle={Proceedings of the 2026 International Conference on Advanced Robotics and Intelligent Systems (ARIS)},
  year={2026},
  note={To appear}
}
```

---
---

# MirrorConv: 參數共享交錯拓撲算子

這份儲存庫包含 **MirrorConv** 的官方 PyTorch 實作。這是一個專為極度輕量化視覺模型設計的卷積算子。

## 🚀 安裝方式

為了最快速地開始使用，您可以直接透過 `pip` 從 GitHub 安裝 MirrorConv。請選擇以下其中一種方式：

**方法一：使用 git 安裝（推薦）**
```bash
pip install git+[https://github.com/chihsunchang1225-design/MirrorConv.git](https://github.com/chihsunchang1225-design/MirrorConv.git)
```

**方法二：直接下載安裝（如果您尚未安裝 git）**
```bash
pip install [https://github.com/chihsunchang1225-design/MirrorConv/archive/refs/heads/main.zip](https://github.com/chihsunchang1225-design/MirrorConv/archive/refs/heads/main.zip)
```

*（註：強烈建議在虛擬環境中進行安裝。）*

## 💻 快速開始

MirrorConv 提供多種「參數共享交錯拓撲算子」的變體。您可以將其當作標準空間卷積層的替代品，輕鬆整合進任何現有的 PyTorch 模型中。

### 基礎用法（核心推薦）
針對極輕量化視覺模型，論文首推使用 **`MirrorConv_Twin`** 算子。其 API 設計高度貼合原生的 `torch.nn.Conv2d`。

```python
import torch
from MirrorConv import MirrorConv_Twin

# 1. 帶入參數初始化算子
layer = MirrorConv_Twin(
    in_channels=32, 
    out_channels=64, 
    kernel_size=3, 
)

# 2. 建立測試輸入張量 (Batch, Channels, Height, Width)
x = torch.randn(1, 32, 224, 224)

# 3. 執行前向傳播
output = layer(x)

print(f"輸入維度: {x.shape}")
print(f"輸出維度: {output.shape}")
# 預期輸出: torch.Size([1, 64, 224, 224])
```

---

## 📚 詳細 API 參考與使用說明

根據您的實驗需求，MirrorConv 提供從高度最佳化的算子到獨立基礎構件等多種模組。

### 1. 核心推薦變體（無動態整流器）
**🟢 極度推薦用於邊緣裝置。** 這是最高效的變體，也是論文實驗最終推薦的設定。

*   `MirrorConv_Twin`: 論文首推算子（雙生模式）。
*   `MirrorConv_Chain`: 論文次推算子（鏈式模式）。
*   `MirrorConv_Random`: 一般輕量化基準（隨機模式）。

**使用範例：**
```python
from MirrorConv import MirrorConv_Twin

# Twin, Chain, Random (無整流器版本) 的呼叫方式與參數完全一致
layer = MirrorConv_Twin(
    in_channels=32,             # [必填] 輸入通道數 (int)
    out_channels=64,            # [必填] 輸出通道數 (int)
    kernel_size=3,              # [選填] 預設: 3
    stride=1,                   # [選填] 預設: 1
    padding=None,               # [選填] 預設: None (當為 None 時，會自動計算並保持空間大小)
    dilation=1,                 # [選填] 預設: 1
    bias=False                  # [選填] 預設: False
)
```

### 2. 探索與消融實驗變體（含動態整流器）
**🟠 供研究者進行邊界探索使用。** 這些變體包含了動態整流區塊（Dynamic Rectification）。論文的實驗結果證明這並非在所有情況下都是最佳設定，但非常適合用於消融實驗（Ablation Study）。

*   `MirrorConv_Twin_Rectifier`
*   `MirrorConv_Chain_Rectifier`
*   `MirrorConv_Random_Rectifier`

**使用範例：**
```python
from MirrorConv import MirrorConv_Twin_Rectifier

# 包含動態整流器的變體，額外提供 'rectifier_kernel_size' 參數
layer_with_rectifier = MirrorConv_Twin_Rectifier(
    in_channels=32,             # [必填] 輸入通道數
    out_channels=64,            # [必填] 輸出通道數
    kernel_size=3,              # [選填] 預設: 3
    stride=1,                   # [選填] 預設: 1
    padding=None,               # [選填] 預設: None
    dilation=1,                 # [選填] 預設: 1
    bias=False,                 # [選填] 預設: False
    rectifier_kernel_size=3     # [選填] 整流器專用的卷積核大小。若為 None，則預設等同於 kernel_size
)
```

### 3. 基礎核心算子
如果您想要透過參數直接控制所有的底層架構行為（例如手動切換 Twin/Chain 或整流器狀態），可以直接呼叫 `Base_MirrorConv`。

```python
from MirrorConv import Base_MirrorConv

base_layer = Base_MirrorConv(
    in_channels=32,
    out_channels=64,
    kernel_size=3,
    
    # --- Base_MirrorConv 專屬架構控制參數 ---
    mode='standard',            # 模式控制字串 (預設: 'standard')
                                # 支援: 'standard', 'full_random', 'standard_no_Rectifier', 'full_random_no_Rectifier'
    twin_mode=True              # 通道奇偶數分組邏輯 (預設: True)
                                # True 代表 Twin 模式邏輯，False 代表 Chain 模式邏輯
)
```

### 4. 內部子模組（供進階自定義）
為了方便進行深度的特徵提取與自定義架構設計，我們將底層的獨立構件對外開放。

**區塊 A: MirrorFusion (特徵融合與對稱生成)**
```python
from MirrorConv import MirrorFusion

fusion_block = MirrorFusion(
    in_channels=32,             # [必填] 輸入通道數
    half_channels=16,           # [必填] 單側特徵通道數 (通常為 out_channels 的一半)
    aligned_out=32,             # [必填] 對齊後的總輸出通道數 (通常為 half_channels * 2)
    is_random=False             # [必填] False: Mirror模式 / True: Random模式
)
fused_features = fusion_block(torch.randn(1, 32, 224, 224))
```

**區塊 B: Dynamic_Rectifier (動態整流)**
```python
from MirrorConv import Dynamic_Rectifier

rectifier_block = Dynamic_Rectifier(
    aligned_out=32,             # [必填] 來自 Fusion 層的對齊輸出通道數
    rectifier_kernel_size=3,    # [必填] 整流器使用的空間卷積核大小
    padding=1                   # [必填] 填充大小 (通常為 rectifier_kernel_size // 2)
)
# 輸入建議是已經過 Fusion 處理的特徵
adjusted_features = rectifier_block(fused_features)
```

## 📖 引用

如果您在研究中使用了 MirrorConv，我們強烈建議您優先引用我們的期刊論文（目前審稿中）。若您特別參考了我們早期的發表內容，亦可引用 ARIS 2026 的研討會論文。

**【優先引用】期刊論文（審稿中）：**
> 核心架構與完整實驗目前已投稿至期刊，待正式接收後將於此處更新引用資訊。

**【次要引用】研討會論文（ARIS 2026）：**
```bibtex
@inproceedings{2026mirrorconv_conf,
  title={MirrorConv: A Parameter-Shared Staggered Topology Operator for Extreme Lightweight Vision Models and its Empirical Robustness Analysis},
  author={Shih, Jia-Shing and Chang, Chi-Hsun},
  booktitle={Proceedings of the 2026 International Conference on Advanced Robotics and Intelligent Systems (ARIS)},
  year={2026},
  note={To appear}
}
```
