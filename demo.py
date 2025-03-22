import torch
import torch.nn as nn
import torch.optim as optim
from build_decoder_transformer import DecoderOnlyTransformer

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Hyperparameters
    vocab_size = 100  # Small vocab for demo
    d_model = 128     # Smaller model for demo
    num_layers = 2    # Fewer layers for demo
    num_heads = 4
    d_ff = 512
    max_len = 50
    dropout = 0.1
    batch_size = 16
    seq_len = 10
    num_epochs = 10
    learning_rate = 0.0005
    
    # Create model
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create a simple sequence prediction task
    # For this demo, we'll create sequences where each token is (position % 10)
    # The model should learn this pattern
    
    # Create training dataset
    def create_batch():
        # Input: [0,1,2,3,4,5,6,7,8,9,0,1,...]
        # Target (shifted by 1): [1,2,3,4,5,6,7,8,9,0,1,2,...]
        inputs = torch.tensor([[i % 10 for i in range(j, j + seq_len)] 
                              for j in range(batch_size)])
        targets = torch.tensor([[i % 10 for i in range(j + 1, j + seq_len + 1)] 
                               for j in range(batch_size)])
        return inputs, targets
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Create new batch for this epoch
        inputs, targets = create_batch()
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)  # shape: (batch_size, seq_len, vocab_size)
        
        # Reshape for loss calculation
        outputs = outputs.view(-1, vocab_size)  # (batch_size*seq_len, vocab_size)
        targets = targets.view(-1)              # (batch_size*seq_len)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
    
    # Test the model with inference
    print("\nTesting the model...")
    model.eval()
    
    # Create a test sequence: [0,1,2,3,4]
    test_seq = torch.tensor([[0, 1, 2, 3, 4]])
    
    # Generate 5 more tokens
    for _ in range(5):
        with torch.no_grad():
            # Get prediction
            output = model(test_seq)
            # Get next token prediction (last token in sequence)
            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            # Append to sequence
            test_seq = torch.cat([test_seq, next_token], dim=1)
    
    print(f"Input: [0,1,2,3,4]")
    print(f"Generated sequence: {test_seq.tolist()[0]}")
    print(f"Expected pattern: [0,1,2,3,4,5,6,7,8,9]")

if __name__ == "__main__":
    main() 