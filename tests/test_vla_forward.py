import torch
from src.models.vla_model import BitMambaVLA

def test_vla_forward():
    print("🧪 Testing BitMambaVLA Forward Pass...")
    
    batch_size = 2
    action_dim = 7   # 例: 6自由度アーム + 1グッパー
    chunk_size = 5   # 5ステップ先まで予測
    
    # ダミー画像入力 (Batch, Channel, Height, Width)
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    # ダミーテキスト埋め込み (Batch, Seq_len, d_model)
    dummy_text = torch.randn(batch_size, 8, 1024)
    
    # VLAモデルの初期化 (テスト用に軽量構成)
    vla = BitMambaVLA(
        vision_model_name="vit_tiny_patch16_224", # テスト用に軽量モデル指定
        d_model=1024,
        n_layers=2,
        n_heads=16,
        action_dim=action_dim,
        chunk_size=chunk_size,
        pretrained_vision=False
    )
    
    # 順伝播の実行
    actions = vla(pixel_values=dummy_images, text_embeds=dummy_text)
    
    print(f"  Input Image Shape: {dummy_images.shape}")
    print(f"  Output Action Shape: {actions.shape}")
    
    assert actions.shape == (batch_size, chunk_size, action_dim), f"Expected shape {(batch_size, chunk_size, action_dim)}, but got {actions.shape}"
    print("✅ BitMambaVLA Forward OK!")

if __name__ == "__main__":
    test_vla_forward()