"""
Test SSM setup_optimizers function. Example run:

python test_ssm_setup_optimizers.py
"""

import torch
from nanochat.ssm import SSM, SSMConfig

def test_setup_optimizers():
    """Test that setup_optimizers runs correctly on a small SSM model."""
    
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
    print("Creating SSM model...")
    model = SSM(config)
    
    # Initialize weights
    print("Initializing weights...")
    model.init_weights()
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal model parameters: {total_params:,}")
    
    # Test setup_optimizers
    print("\n" + "="*60)
    print("Testing setup_optimizers...")
    print("="*60 + "\n")
    
    try:
        optimizers = model.setup_optimizers(
            unembedding_lr=0.004,
            embedding_lr=0.2,
            matrix_lr=0.02,
            weight_decay=0.0
        )
        
        print("\n" + "="*60)
        print("setup_optimizers completed successfully!")
        print("="*60)
        
        # Verify we got the expected optimizers
        assert len(optimizers) == 2, f"Expected 2 optimizers, got {len(optimizers)}"
        print(f"\n✓ Got {len(optimizers)} optimizers as expected")
        
        # Check optimizer types
        print(f"✓ Optimizer 1 type: {type(optimizers[0]).__name__}")
        print(f"✓ Optimizer 2 type: {type(optimizers[1]).__name__}")
        
        # Verify parameter groups
        for i, opt in enumerate(optimizers):
            print(f"\nOptimizer {i+1} has {len(opt.param_groups)} parameter group(s):")
            for j, group in enumerate(opt.param_groups):
                num_params = sum(p.numel() for p in group['params'])
                print(f"  Group {j+1}: {num_params:,} parameters, lr={group['lr']:.6f}")
        
        print("\n✓ All checks passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error in setup_optimizers: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_setup_optimizers()
    exit(0 if success else 1)

