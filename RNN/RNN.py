import torch

# We define the dataset class here
class RNNDataset(torch.utils.data.Dataset):

    # The max prompt and completion lengths are not super long (usually much
    # less than 50 tokens)
    def __init__(self, data, sp_model, max_prompt_length=50, max_completion_length=10):
        self.data = data
        self.sp = sp_model
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length

    # The length of the dataset is defined just by the number of
    # lines in it
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # We encode the prompt and completion into token ids
        # on-demand as batches are created
        prompt_ids = self.sp.encode(item["prompt"], out_type=int)[-self.max_prompt_length:]
        
        completion_ids = self.sp.encode(item["completion"], out_type=int)[:self.max_completion_length]

        # Input IDs contains everything except the last token
        input_ids = torch.tensor(prompt_ids + completion_ids[:-1])

        # Target IDs masks everything but the completion part
        target_ids = torch.tensor([-100]*len(prompt_ids) + completion_ids[1:])

        return input_ids, target_ids


# Function used for DataLoader class 
def collate_fn(batch):

    # Convert the batch into two lists of input and target sequences
    input_seqs, target_seqs = zip(*batch)

    # Pad the input using the pad value (integer 3)
    input_padded = torch.nn.utils.rnn.pad_sequence(input_seqs, batch_first=True, padding_value = 3)
    
    # Target is padded using mask value (integer -100)
    target_padded = torch.nn.utils.rnn.pad_sequence(target_seqs, batch_first=True, padding_value = -100)

    return input_padded, target_padded

"""
Defines an RNN that operates over language to autoregressively
predict the next token from the ones that have come before 
(storing the previous information into a hidden state)
"""
class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        # Embedding layer
        self.embedding = torch.nn.Embedding(num_embeddings=10000, embedding_dim=200, padding_idx=3)

        # Recurrent layers
        self.rnn = torch.nn.RNN(input_size=200, hidden_size=200, num_layers=3, batch_first=True)

        # Passthrough to make a distribution
        self.fc = torch.nn.Linear(in_features=200,out_features=10000)

    # During autoregression, we keep track of hidden state and repeatedly
    # pass it into the forward method
    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        output, hidden = self.rnn(embeds)
        logits = self.fc(output)

        return logits, hidden
    
    def evaluate_model(self, model, validation_loader, loss_func):
        model.eval()
        total_loss = 0

        # Don't waste time computing the gradients during eval step
        # essentially
        with torch.no_grad():
            for x_true, y_true in validation_loader:

                # Validation data is in same device as model
                x_true = x_true.to(next(model.parameters()).device)
                y_true = y_true.to(next(model.parameters()).device)

                # Predict y output from dataloader
                y_pred, _ = model(x_true)

                loss = loss_func(
                    # We need to flatten the output of the predicted
                    # classes of the predicted output against the
                    # true output
                    y_pred.view(-1, y_pred.size(-1)),
                    y_true.view(-1)
                )

                # Add loss for this batch to total loss for the epoch
                total_loss += loss.item()
            
        # Returns total loss produced by the entire validation set.
        return total_loss

    def train_model(self, model, train_loader, validation_loader, optimizer=None, start_epoch=0, epochs=5, lr=1e-3):

        # We define -100 to be the ignore index, as this integer
        # is used to denote masked tokens in the target 
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)

        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2)

        for epoch in range(start_epoch, epochs):

            # Switch model back to training mode after each eval
            # after the end of each epoch
            model.train()

            current_loss = 0

            for x_true, y_true in train_loader:

                # Batch data is in same device as model
                x_true = x_true.to(next(model.parameters()).device)
                y_true = y_true.to(next(model.parameters()).device)

                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"NaN detected in gradients of {name}")


                optimizer.zero_grad()

                # Predict y output from dataloader
                # (During training we throw out the hidden state)
                y_pred, _ = model(x_true)

                loss = loss_func(
                    # We need to flatten the output of the predicted
                    # classes of the predicted output against the
                    # true output
                    y_pred.view(-1, y_pred.size(-1)),
                    y_true.view(-1)
                )

                print("Training Loss : ",loss)

                # Backpropagate Loss, to update parameters in the model
                loss.backward()

                # Gradient clipping is done to stabilize training in RNNs
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Advance optimizer according to learning rate
                optimizer.step()

                # This is kept for record keeping and printing later
                current_loss = loss
            
            # Run evaluation
            validation_loss = self.evaluate_model(model=model, validation_loader=validation_loader, loss_func=loss_func)

            # Advance learning rate scheduler (decrease on plateau), using the validation loss.
            scheduler.step(validation_loss / len(validation_loader))

            # Save checkpoint
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'validation_loss': validation_loss,
            }, f"checkpoints/checkpoint_epoch_{epoch+1}_{validation_loss/len(validation_loader)}.pt")

            print(f"Epoch {epoch+1}, Training Loss: {current_loss:.4f}, Validation Loss: {validation_loss / len(validation_loader):.4f}")
    
    @torch.no_grad()
    def generate(self, prompt, sp_model, max_new_tokens=20, device="cpu"):
        self.eval()

        prompt_ids = sp_model.encode(prompt, out_type=int)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)

        # Get initial hidden state
        _, hidden = self.forward(prompt_tensor)

        generated = prompt_ids[:]

        for _ in range(max_new_tokens):

            # Get the last token which has been generated (starting from the end of the
            # prompt and put it into memory
            last_token = torch.tensor([[generated[-1]]], dtype=torch.long).to(device)

            # Get the output and hidden state from the last token and the previous
            # hidden state
            output, hidden = self.forward(last_token, hidden)

            # Flatten distribution output provided by RNN (shows the likely next tokens)
            next_token_output = output[:, -1, :] / 1.3 # temperature is 1.5

            # Run this distribution through softmax and sample for the next
            # token
            probs = torch.softmax(next_token_output, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token_id)

            print(next_token_id, sp_model.decode([next_token_id]))

            # If the next token ID is eos, then break
            if(next_token_id) == 2:
                break
        
        return sp_model.decode(generated)

# Function to load a checkpoint after we perform a full training run
def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['validation_loss']

# Main function.
if __name__ == "__main__":

    # Load our BPE tokenizer to tokenize our data
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load('../tokenizer/m.model')

    # Load our data into a single object
    import json

    data = []

    with open("../data/train.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))

    # Split data into training and validation sets
    split_point = int(0.8 * len(data))
    train_data = data[:split_point]
    validation_data = data[split_point:]

    # Convert raw data into datasets
    train_dataset = RNNDataset(data=train_data, sp_model=sp)
    validation_dataset = RNNDataset(data=validation_data, sp_model=sp)


    # Convert datasets into data loaders to be consumed by model
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=32, 
        shuffle=False, # No need to shuffle validation data
        collate_fn=collate_fn
    )


    # Set up for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNN()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    TRAIN_RNN = False

    USE_CHECKPOINT = True

    model = RNN()

    if(USE_CHECKPOINT):
        path = "checkpoints/checkpoint_epoch_1_791.9348198771477.pt"

        start_epoch, validation_loss = load_checkpoint(model, optimizer, path)

        print(f"Loading old checkpoint with Validation Loss {validation_loss:.4f}.")

    if(TRAIN_RNN):
        model.train_model(model, train_loader, validation_loader, optimizer, start_epoch)

    print(model.embedding.num_embeddings)

    print(model.generate(prompt="Which do you prefer? Dogs or cats? ",sp_model=sp))

