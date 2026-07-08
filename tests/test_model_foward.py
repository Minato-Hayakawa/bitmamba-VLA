import torch
from src.models.bit_linear import BitLinear
from src.models.bit_mamba2 import BitMamba2LM

def test_forward():
    batch_size = 2
    seq_len = 16
    vocab_size = 1000
    d_model = 128
    n_layers = 2
    n_heads = 4

    print("🧪 Testing BitLinear...")
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    linear = BitLinear(d_model, d_model)
    out = linear(x)
    out.sum().backward()
    assert out.shape == (batch_size, seq_len, d_model)
    print("✅ BitLinear Forward & Backward OK!")

    print("🧪 Testing BitMamba2LM...")
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    model = BitMamba2LM(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads)
    logits = model(input_ids)
    assert logits.shape == (batch_size, seq_len, vocab_size)
    print("✅ BitMamba2LM Forward OK!")

if __name__ == "__main__":
    test_forward()