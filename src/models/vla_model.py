import torch
import torch.nn as nn
import timm
from .bit_linear import BitLinear, RMSNorm
from .bit_mamba2 import BitMambaBlock

class VisionProjector(nn.Module):

    def __init__(self, d_vision: int, d_model: int):
        super().__init__()
        self.proj = BitLinear(d_vision, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
    
class ActionHead(nn.Module):
    """
    BitMambaの出力からロボットの連続アクション(位置・姿勢・グリッパー等)を出力するヘッド
    Action Chunking (複数ステップ先の予測) に対応
    """
    def __init__(self, d_model: int, action_dim: int, chunk_size: int = 1):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        
        # 2層のBitLinear + RMSNorm でアクションを予測
        self.net = nn.Sequential(
            BitLinear(d_model, d_model),
            nn.SiLU(),
            BitLinear(d_model, action_dim * chunk_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, d_model) -> 最後のトークンまたは統合表現
        out = self.net(x)
        # (Batch, chunk_size, action_dim) に整形
        batch_size = x.shape[0]
        return out.view(batch_size, self.chunk_size, self.action_dim)
    
class BitMambaVLA(nn.Module):
    """
    Vision Encoder + Vision Projector + BitMamba Backbone + Action Head
    """
    def __init__(
        self,
        vision_model_name: str = "vit_base_patch14_dinov2.lvd142m",
        d_model: int = 1024,
        n_layers: int = 12,
        n_heads: int = 16,
        action_dim: int = 7,      # 例: x, y, z, roll, pitch, yaw, gripper (7自由度)
        chunk_size: int = 10,     # Action Chunking: 今後10ステップの動作を一度に予測
        pretrained_vision: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        
        # 1. Vision Encoder (timmを利用)
        self.vision_encoder = timm.create_model(
            vision_model_name,
            pretrained=pretrained_vision,
            num_classes=0  # 特徴量抽出器として使用 (Classification headを削除)
        )
        d_vision = self.vision_encoder.num_features
        
        # 2. Vision Projector
        self.vision_projector = VisionProjector(d_vision=d_vision, d_model=d_model)
        
        # 3. BitMamba Backbone Layers
        self.layers = nn.ModuleList([
            BitMambaBlock(d_model=d_model, n_heads=n_heads)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        
        # 4. Action Head
        self.action_head = ActionHead(d_model=d_model, action_dim=action_dim, chunk_size=chunk_size)

    def forward(self, pixel_values: torch.Tensor, text_embeds: torch.Tensor = None) -> torch.Tensor:
        """
        pixel_values: (Batch, C, H, W) - 画像入力
        text_embeds:  (Batch, Seq_text, d_model) - テキスト指示文の事前埋め込み (任意)
        returns:      (Batch, chunk_size, action_dim) - 予測されたロボット動作
        """
        batch_size = pixel_values.shape[0]
        
        # --- A. 視覚特徴量の抽出 & プロジェクション ---
        # timm の forward_features を使い、パッチトークン列を取得 (Batch, Num_Patches, d_vision)
        v_feat = self.vision_encoder.forward_features(pixel_values)
        if v_feat.ndim == 2:  # グローバルプーリングされている場合は3次元に展開
            v_feat = v_feat.unsqueeze(1)
            
        v_tokens = self.vision_projector(v_feat)  # (Batch, Num_Patches, d_model)
        
        # --- B. 視覚トークン と テキストトークン の結合 ---
        if text_embeds is not None:
            # [Vision Tokens, Text Tokens] を系列方向に結合
            seq = torch.cat([v_tokens, text_embeds], dim=1)
        else:
            seq = v_tokens
            
        # --- C. BitMamba バックボーンを通過 ---
        x = seq
        for layer in self.layers:
            x = x + layer(x)
        x = self.norm_f(x)
        
        # --- D. アクション予測 ---
        # 系列の「最後のトークン」の情報（視覚と指示文がアライメントされた文脈）を取り出して Action Head へ渡す
        last_token_feat = x[:, -1, :]  # (Batch, d_model)
        actions = self.action_head(last_token_feat)  # (Batch, chunk_size, action_dim)
        
        return actions