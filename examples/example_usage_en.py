import torch

# Import all 9 public APIs
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
    print("🚀 Starting complete MirrorConv API test...\n")
    
    # Create a unified test input tensor (Batch=1, Channels=32, Height=224, Width=224)
    x = torch.randn(1, 32, 224, 224)
    print(f"📦 Test input shape: {x.shape}\n")
    print("-" * 50)

    # ==========================================
    # Test Group A: Primary Recommended Variants (No Dynamic Rectifier)
    # ==========================================
    print("🟢 Test Group A: Primary Recommended Variants (No Dynamic Rectifier)")
    
    # 1. MirrorConv_Twin
    layer_twin = MirrorConv_Twin(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
    out_twin = layer_twin(x)
    print(f"✅ [1/9] MirrorConv_Twin output successful -> {out_twin.shape}")

    # 2. MirrorConv_Chain
    layer_chain = MirrorConv_Chain(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True)
    out_chain = layer_chain(x)
    print(f"✅ [2/9] MirrorConv_Chain output successful -> {out_chain.shape}")

    # 3. MirrorConv_Random
    layer_random = MirrorConv_Random(in_channels=32, out_channels=64, kernel_size=5, dilation=2)
    out_random = layer_random(x)
    print(f"✅ [3/9] MirrorConv_Random output successful -> {out_random.shape}\n")

    # ==========================================
    # Test Group B: Exploratory & Ablation Variants (With Dynamic Rectifier)
    # ==========================================
    print("🟠 Test Group B: Exploratory & Ablation Variants (With Dynamic Rectifier)")
    
    # 4. MirrorConv_Twin_Rectifier
    layer_twin_rec = MirrorConv_Twin_Rectifier(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, rectifier_kernel_size=3)
    out_twin_rec = layer_twin_rec(x)
    print(f"✅ [4/9] MirrorConv_Twin_Rectifier output successful -> {out_twin_rec.shape}")

    # 5. MirrorConv_Chain_Rectifier
    layer_chain_rec = MirrorConv_Chain_Rectifier(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True, rectifier_kernel_size=5)
    out_chain_rec = layer_chain_rec(x)
    print(f"✅ [5/9] MirrorConv_Chain_Rectifier output successful -> {out_chain_rec.shape}")

    # 6. MirrorConv_Random_Rectifier
    layer_random_rec = MirrorConv_Random_Rectifier(in_channels=32, out_channels=64, kernel_size=5, dilation=2)
    out_random_rec = layer_random_rec(x)
    print(f"✅ [6/9] MirrorConv_Random_Rectifier output successful -> {out_random_rec.shape}\n")

    # ==========================================
    # Test Group C: Core Base Operator
    # ==========================================
    print("🔵 Test Group C: Core Base Operator")
    
    # 7. Base_MirrorConv (Using custom parameters)
    layer_base = Base_MirrorConv(
        in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, 
        mode='standard_no_Rectifier', twin_mode=True
    )
    out_base = layer_base(x)
    print(f"✅ [7/9] Base_MirrorConv output successful -> {out_base.shape}\n")

    # ==========================================
    # Test Group D: Internal Sub-Modules (Block A & Block B)
    # ==========================================
    print("⚙️ Test Group D: Internal Sub-Modules")
    
    # 8. MirrorFusion (Block A)
    # Parameters setup: in_channels=32, half_channels=16, aligned_out=32
    fusion_block = MirrorFusion(in_channels=32, half_channels=16, aligned_out=32, is_random=False)
    fused_out = fusion_block(x)
    print(f"✅ [8/9] MirrorFusion output successful -> {fused_out.shape}")

    # 9. Dynamic_Rectifier (Block B)
    # Must receive the output from the Fusion block as input
    rectifier_block = Dynamic_Rectifier(aligned_out=32, rectifier_kernel_size=3, padding=1)
    rectifier_out = rectifier_block(fused_out)
    print(f"✅ [9/9] Dynamic_Rectifier output successful -> {rectifier_out.shape}\n")

    print("-" * 50)
    print("🎉 All 9 APIs tested successfully! Package imports and parameter designs are working perfectly.")

if __name__ == "__main__":
    test_mirrorconv_apis()