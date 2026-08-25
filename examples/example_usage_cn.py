import torch

# 匯入所有 9 個公開 API
from MirrorConv import (
    Base_MirrorConv,
    MirrorConv_Twin,
    MirrorConv_Chain,
    MirrorConv_Random,
    MirrorConv_Twin_Rectifier,
    MirrorConv_Chain_Rectifier,
    MirrorConv_Random_Rectifier,
    MirrorFusion,
    Dynamic_Rectifier
)

def test_mirrorconv_apis():
    print("🚀 啟動 MirrorConv API 完整測試...\n")
        
    # 建立統一的測試輸入張量 (Batch=1, Channels=32, Height=224, Width=224)
    x = torch.randn(1, 32, 224, 224)
    print(f"📦 測試輸入維度: {x.shape}\n")
    print("-" * 50)

    # ==========================================
    # 測試組 A: 核心推薦變體 (無動態整流器)
    # ==========================================
    print("🟢 測試組 A: 核心推薦變體 (無動態整流器)")
    
    # 1. MirrorConv_Twin
    layer_twin = MirrorConv_Twin(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
    out_twin = layer_twin(x)
    print(f"✅ [1/9] MirrorConv_Twin 輸出成功 -> {out_twin.shape}")

    # 2. MirrorConv_Chain
    layer_chain = MirrorConv_Chain(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True)
    out_chain = layer_chain(x)
    print(f"✅ [2/9] MirrorConv_Chain 輸出成功 -> {out_chain.shape}")

    # 3. MirrorConv_Random
    layer_random = MirrorConv_Random(in_channels=32, out_channels=64, kernel_size=5, dilation=2)
    out_random = layer_random(x)
    print(f"✅ [3/9] MirrorConv_Random 輸出成功 -> {out_random.shape}\n")

    # ==========================================
    # 測試組 B: 探索與消融實驗變體 (含動態整流器)
    # ==========================================
    print("🟠 測試組 B: 探索與消融實驗變體 (含動態整流器)")
    
    # 4. MirrorConv_Twin_Rectifier
    layer_twin_rec = MirrorConv_Twin_Rectifier(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, rectifier_kernel_size=3)
    out_twin_rec = layer_twin_rec(x)
    print(f"✅ [4/9] MirrorConv_Twin_Rectifier 輸出成功 -> {out_twin_rec.shape}")

    # 5. MirrorConv_Chain_Rectifier
    layer_chain_rec = MirrorConv_Chain_Rectifier(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True, rectifier_kernel_size=5)
    out_chain_rec = layer_chain_rec(x)
    print(f"✅ [5/9] MirrorConv_Chain_Rectifier 輸出成功 -> {out_chain_rec.shape}")

    # 6. MirrorConv_Random_Rectifier
    layer_random_rec = MirrorConv_Random_Rectifier(in_channels=32, out_channels=64, kernel_size=5, dilation=2)
    out_random_rec = layer_random_rec(x)
    print(f"✅ [6/9] MirrorConv_Random_Rectifier 輸出成功 -> {out_random_rec.shape}\n")

    # ==========================================
    # 測試組 C: 基礎核心算子
    # ==========================================
    print("🔵 測試組 C: 基礎核心算子")
    
    # 7. Base_MirrorConv (使用自定義參數)
    layer_base = Base_MirrorConv(
        in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, 
        mode='standard_no_Rectifier', twin_mode=True
    )
    out_base = layer_base(x)
    print(f"✅ [7/9] Base_MirrorConv 輸出成功 -> {out_base.shape}\n")

    # ==========================================
    # 測試組 D: 內部獨立構件 (Block A & Block B)
    # ==========================================
    print("⚙️ 測試組 D: 內部獨立構件")
    
    # 8. MirrorFusion (Block A)
    # 參數設定: in_channels=32, half_channels=16, aligned_out=32
    fusion_block = MirrorFusion(in_channels=32, half_channels=16, aligned_out=32, is_random=False)
    fused_out = fusion_block(x)
    print(f"✅ [8/9] MirrorFusion 輸出成功 -> {fused_out.shape}")

    # 9. Dynamic_Rectifier (Block B)
    # 必須接收來自 Fusion block 的輸出作為輸入
    rectifier_block = Dynamic_Rectifier(aligned_out=32, rectifier_kernel_size=3, padding=1)
    rectifier_out = rectifier_block(fused_out)
    print(f"✅ [9/9] Dynamic_Rectifier 輸出成功 -> {rectifier_out.shape}\n")

    print("-" * 50)
    print("🎉 所有 9 個 API 測試完畢，套件讀取與參數設計皆正常運作！")

if __name__ == "__main__":
    test_mirrorconv_apis()