"""
Test SSM midtraining. This script:
1. Creates a minimal SSM base model checkpoint (if needed)
2. Runs midtraining with minimal iterations
3. Verifies the checkpoint is saved correctly with model_type="ssm"
4. Verifies the checkpoint can be loaded back

Example run:
python test_ssm_midtrain.py
"""

import os
import torch

from nanochat.ssm import SSM, SSMConfig
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.common import get_base_dir, print0
from nanochat.tokenizer import get_tokenizer


def create_minimal_ssm_base_checkpoint(device, base_dir, depth=2):
    """Create a minimal SSM base model checkpoint for testing."""
    
    print("="*60)
    print("Creating minimal SSM base model checkpoint")
    print("="*60 + "\n")
    
    # Get the actual tokenizer vocab size
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    
    # Create a small SSM config for testing
    max_seq_len = 128
    model_dim = depth * 64
    
    config = SSMConfig(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_embd=model_dim,
        ssm_state_dim=min(64, model_dim),
        ssm_conv_kernel=4,
        expand_factor=2,
    )
    
    print(f"SSM Config:")
    print(f"  sequence_len: {config.sequence_len}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  n_layer: {config.n_layer}")
    print(f"  n_embd: {config.n_embd}")
    print(f"  ssm_state_dim: {config.ssm_state_dim}")
    print(f"  ssm_conv_kernel: {config.ssm_conv_kernel}")
    print(f"  expand_factor: {config.expand_factor}\n")
    
    # Create the model
    print("Creating SSM model...")
    with torch.device("meta"):
        model = SSM(config)
    
    model.to_empty(device=device)
    model.init_weights()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")
    
    # Create dummy optimizers (needed for checkpoint format)
    optimizers = model.setup_optimizers(
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0
    )
    
    # Save checkpoint
    output_dirname = f"ssm_d{depth}"
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    step = 0
    model_config_kwargs = {
        "sequence_len": max_seq_len,
        "vocab_size": vocab_size,
        "n_layer": depth,
        "n_embd": model_dim,
        "ssm_state_dim": config.ssm_state_dim,
        "ssm_conv_kernel": config.ssm_conv_kernel,
        "expand_factor": config.expand_factor,
    }
    
    print(f"Saving checkpoint to: {checkpoint_dir}")
    save_checkpoint(
        checkpoint_dir,
        step,
        model.state_dict(),
        [opt.state_dict() for opt in optimizers],
        {
            "step": step,
            "val_bpb": 2.5,  # dummy value
            "model_type": "ssm",
            "model_config": model_config_kwargs,
            "user_config": {},
            "device_batch_size": 1,
            "max_seq_len": max_seq_len,
        },
        rank=0,
    )
    
    print(f"✓ Checkpoint saved successfully\n")
    return checkpoint_dir, output_dirname


def test_ssm_midtrain():
    """Test SSM midtraining end-to-end."""
    
    print("="*60)
    print("Testing SSM Midtraining")
    print("="*60 + "\n")
    
    # Check if CUDA is available (mamba_ssm requires CUDA)
    if not torch.cuda.is_available():
        print("⚠ Warning: CUDA is not available.")
        print("mamba_ssm requires CUDA, so this test will be skipped.")
        print("To run this test, you need a CUDA-capable GPU.")
        return True  # Skip gracefully
    
    device = torch.device("cuda")
    base_dir = get_base_dir()
    
    # Step 1: Create or verify base checkpoint exists
    print("Step 1: Setting up base checkpoint...")
    depth = 2
    model_tag = f"ssm_d{depth}"
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", model_tag)
    
    if not os.path.exists(checkpoint_dir) or not os.path.exists(
        os.path.join(checkpoint_dir, "model_000000.pt")
    ):
        print("  Base checkpoint not found, creating one...")
        create_minimal_ssm_base_checkpoint(device, base_dir, depth=depth)
    else:
        print(f"  ✓ Base checkpoint already exists at {checkpoint_dir}")
    
    # Step 2: Verify we can load the base model
    print("\nStep 2: Verifying base model can be loaded...")
    try:
        model, tokenizer, meta = load_model("base", device, phase="train", model_tag=model_tag, step=0)
        model_type = meta.get("model_type", "gpt")
        print(f"  ✓ Base model loaded successfully")
        print(f"  ✓ Model type: {model_type}")
        
        if model_type != "ssm":
            print(f"  ✗ ERROR: Expected model_type='ssm', got '{model_type}'")
            return False
        
        # Verify SSM-specific config fields
        model_config = meta["model_config"]
        required_ssm_fields = ["ssm_state_dim", "ssm_conv_kernel", "expand_factor"]
        for field in required_ssm_fields:
            if field not in model_config:
                print(f"  ✗ ERROR: Missing SSM config field: {field}")
                return False
            print(f"  ✓ {field}: {model_config[field]}")
        
    except Exception as e:
        print(f"  ✗ ERROR: Failed to load base model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Run midtraining with minimal settings
    print("\nStep 3: Running midtraining (minimal test run)...")
    print("  This will run a few iterations to verify the training loop works.\n")
    
    # Import midtraining components
    from collections import deque
    from nanochat.common import compute_init, compute_cleanup, DummyWandb
    from nanochat.tokenizer import get_token_bytes
    from nanochat.loss_eval import evaluate_bpb
    from nanochat.checkpoint_manager import save_checkpoint as save_checkpoint_fn
    
    # Minimal training settings
    device_batch_size = 1
    max_seq_len = 128
    num_iterations = 3  # Just a few iterations for testing
    total_batch_size = device_batch_size * max_seq_len
    eval_every = -1  # Disable eval for speed
    
    # Setup DDP (single GPU for testing)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init("cuda")
    master_process = ddp_rank == 0
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    synchronize = torch.cuda.synchronize
    
    # Load model again for training
    model, tokenizer, meta = load_model("base", device, phase="train", model_tag=model_tag, step=0)
    model_type = meta.get("model_type", "gpt")
    assert model_type == "ssm", f"Expected SSM model, got {model_type}"
    
    orig_model = model
    model = torch.compile(model, dynamic=False)
    
    # Setup optimizers
    optimizers = model.setup_optimizers(
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0
    )
    
    # Create minimal training data (just dummy data for testing)
    print("  Creating minimal training data...")
    vocab_size = tokenizer.get_vocab_size()
    
    def dummy_data_generator():
        """Generate dummy training data with valid token IDs."""
        for _ in range(num_iterations + 1):  # +1 for prefetch
            # Use valid token IDs (0 to vocab_size-1)
            inputs = torch.randint(0, vocab_size, (device_batch_size, max_seq_len), dtype=torch.int32, device=device)
            targets = torch.randint(0, vocab_size, (device_batch_size, max_seq_len), dtype=torch.int64, device=device)
            yield inputs, targets
    
    train_loader = dummy_data_generator()
    x, y = next(train_loader)  # Prefetch first batch
    
    # Training loop (minimal)
    print("  Running training iterations...")
    step = 0
    last_step = False
    
    try:
        while step < num_iterations:
            if step == num_iterations - 1:
                last_step = True
            
            # Forward and backward
            synchronize()
            with autocast_ctx:
                loss = model(x, y)
            loss.backward()
            
            # Optimizer step
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
            synchronize()
            
            loss_val = loss.item()
            print(f"    Step {step+1}/{num_iterations}: loss = {loss_val:.6f}")
            
            step += 1
            if not last_step:
                x, y = next(train_loader)
        
        print("  ✓ Training loop completed successfully\n")
        
    except Exception as e:
        print(f"  ✗ ERROR: Training loop failed: {e}")
        import traceback
        traceback.print_exc()
        compute_cleanup()
        return False
    
    # Step 4: Save checkpoint and verify
    print("Step 4: Saving midtraining checkpoint...")
    try:
        output_dirname = f"{model_type}_d{depth}"
        mid_checkpoint_dir = os.path.join(base_dir, "mid_checkpoints", output_dirname)
        os.makedirs(mid_checkpoint_dir, exist_ok=True)
        
        # Build model_config
        model_config = {
            "sequence_len": max_seq_len,
            "vocab_size": tokenizer.get_vocab_size(),
            "n_layer": depth,
            "n_embd": model.config.n_embd,
            "ssm_state_dim": model.config.ssm_state_dim,
            "ssm_conv_kernel": model.config.ssm_conv_kernel,
            "expand_factor": model.config.expand_factor,
        }
        
        val_bpb = 2.5  # Dummy value for testing
        
        if master_process:
            save_checkpoint_fn(
                mid_checkpoint_dir,
                step,
                orig_model.state_dict(),
                [opt.state_dict() for opt in optimizers],
                {
                    "step": step,
                    "val_bpb": val_bpb,
                    "model_type": model_type,
                    "model_config": model_config,
                    "user_config": {},
                },
                rank=ddp_rank,
            )
            print(f"  ✓ Checkpoint saved to: {mid_checkpoint_dir}")
        
    except Exception as e:
        print(f"  ✗ ERROR: Failed to save checkpoint: {e}")
        import traceback
        traceback.print_exc()
        compute_cleanup()
        return False
    
    compute_cleanup()
    
    # Step 5: Verify checkpoint can be loaded
    print("\nStep 5: Verifying midtraining checkpoint can be loaded...")
    try:
        loaded_model, loaded_tokenizer, loaded_meta = load_model(
            "mid", device, phase="eval", model_tag=output_dirname, step=step
        )
        loaded_model_type = loaded_meta.get("model_type", "gpt")
        
        print(f"  ✓ Checkpoint loaded successfully")
        print(f"  ✓ Model type: {loaded_model_type}")
        
        if loaded_model_type != "ssm":
            print(f"  ✗ ERROR: Expected model_type='ssm', got '{loaded_model_type}'")
            return False
        
        # Verify SSM config fields are present
        loaded_config = loaded_meta["model_config"]
        for field in ["ssm_state_dim", "ssm_conv_kernel", "expand_factor"]:
            if field not in loaded_config:
                print(f"  ✗ ERROR: Missing SSM config field in checkpoint: {field}")
                return False
            print(f"  ✓ {field}: {loaded_config[field]}")
        
        # Verify checkpoint directory name includes model_type
        if not output_dirname.startswith("ssm_"):
            print(f"  ✗ ERROR: Checkpoint directory should start with 'ssm_', got '{output_dirname}'")
            return False
        print(f"  ✓ Checkpoint directory name is correct: {output_dirname}")
        
    except Exception as e:
        print(f"  ✗ ERROR: Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
    print("\nSSM midtraining is working correctly!")
    print(f"  - Base model checkpoint: {checkpoint_dir}")
    print(f"  - Midtraining checkpoint: {mid_checkpoint_dir}")
    print(f"  - Model type correctly saved and loaded: {model_type}")
    
    return True


if __name__ == "__main__":
    success = test_ssm_midtrain()
    exit(0 if success else 1)

