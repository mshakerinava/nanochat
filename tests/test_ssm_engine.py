"""
Test SSM model with Engine for inference. Example run:

python test_ssm_engine.py
"""

import torch
from nanochat.ssm import SSM, SSMConfig
from nanochat.engine import Engine


class SimpleTokenizer:
    """Simple mock tokenizer for testing."""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.bos_token_id = 0
        self.special_tokens = {
            "<|bos|>": 0,
            "<|python_start|>": 1,
            "<|python_end|>": 2,
            "<|output_start|>": 3,
            "<|output_end|>": 4,
            "<|assistant_end|>": 5,
        }
    
    def encode(self, text, prepend=None):
        """Simple encoding: convert text to list of token IDs."""
        # For testing, just return a simple sequence
        if isinstance(text, str):
            # Convert string to token IDs (simple mapping)
            tokens = [ord(c) % (self.vocab_size - 10) + 10 for c in text[:20]]  # Limit length
        else:
            tokens = text
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens
    
    def decode(self, token_ids):
        """Simple decoding: convert token IDs back to text."""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        # Simple reverse mapping
        return "".join([chr(t % 256) if t < 256 else "?" for t in token_ids])
    
    def encode_special(self, special_token):
        """Encode a special token."""
        return self.special_tokens.get(special_token, 0)
    
    def get_bos_token_id(self):
        """Get the BOS token ID."""
        return self.bos_token_id


def test_ssm_engine_inference():
    """Test SSM model inference using the Engine."""
    
    print("="*60)
    print("Testing SSM model with Engine")
    print("="*60 + "\n")
    
    # Create a small SSM config for testing
    config = SSMConfig(
        sequence_len=128,
        vocab_size=1000,
        n_layer=2,
        n_embd=128,
        ssm_state_dim=64,
        ssm_conv_kernel=4,
        expand_factor=2,
    )
    
    # Create the model
    print("1. Creating SSM model...")
    model = SSM(config)
    
    # Initialize weights
    print("2. Initializing weights...")
    model.init_weights()
    
    # Move model to device (mamba_ssm requires CUDA)
    if not torch.cuda.is_available():
        print("   ⚠ Warning: CUDA is not available. mamba_ssm requires CUDA.")
        print("   Skipping test (mamba_ssm does not support CPU inference).")
        return True  # Skip test gracefully
    
    device = torch.device("cuda")
    model = model.to(device)
    print(f"   Model moved to device: {device}")
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}\n")
    
    # Create a simple tokenizer
    print("3. Creating tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    
    # Create Engine with SSM model
    print("4. Creating Engine with SSM model...")
    engine = Engine(model, tokenizer, model_type="ssm")
    print("   ✓ Engine created successfully\n")
    
    # Test inference
    print("5. Testing inference...")
    prompt = "Hello world"
    prompt_tokens = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
    print(f"   Prompt: '{prompt}'")
    print(f"   Prompt tokens: {prompt_tokens}\n")
    
    # Generate tokens using the engine
    print("6. Generating tokens...")
    max_tokens = 10
    generated_tokens = []
    try:
        for i, (token_column, token_masks) in enumerate(
            engine.generate(
                prompt_tokens,
                num_samples=1,
                max_tokens=max_tokens,
                temperature=1.0,
                top_k=None,
                seed=42
            )
        ):
            token = token_column[0]
            mask = token_masks[0]
            generated_tokens.append(token)
            decoded = tokenizer.decode([token])
            print(f"   Step {i+1}: token={token}, mask={mask}, decoded='{decoded}'")
            if i >= max_tokens - 1:
                break
        
        print(f"\n   ✓ Generated {len(generated_tokens)} tokens successfully")
        print(f"   Generated token sequence: {generated_tokens}")
        
        # Test batch generation
        print("\n7. Testing batch generation (num_samples=2)...")
        results, masks = engine.generate_batch(
            prompt_tokens,
            num_samples=2,
            max_tokens=5,
            temperature=1.0,
            seed=42
        )
        print(f"   ✓ Generated {len(results)} samples")
        for i, (result, mask) in enumerate(zip(results, masks)):
            print(f"   Sample {i+1}: {len(result)} tokens, {sum(mask)} sampled, {len(mask) - sum(mask)} forced")
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_ssm_engine_inference()
    exit(0 if success else 1)

