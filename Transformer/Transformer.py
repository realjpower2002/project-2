import torch

# We define the dataset class here
class TransformerDataset(torch.utils.data.Dataset):

    # The max prompt and completion lengths are not super long (usually much
    # less than 50 tokens)
    def __init__(self, data, sp_model, block_size=512):
        self.data = data
        self.sp = sp_model
        self.block_size=block_size

        all_text = [item["prompt"] + " " + item["completion"] for item in data]
        tokens = []
        for text in all_text:
            tokens.extend(sp_model.encode(text, out_type=int))

        # Store as a flat tensor
        self.tokens = torch.tensor(tokens, dtype=torch.long)

    # The length of the dataset is defined just by the number of
    # lines in it
    def __len__(self):
        return max(0, len(self.tokens) - self.block_size)


    def __getitem__(self, idx):
        # Feed in x and y as idx, idx+1 pairs. These are passed into
        # the model on its whole length, and each position tries to
        # predict the next token in the sequence
        x = self.tokens[idx : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + self.block_size + 1]

        return x, y


# Function used for DataLoader class 
def collate_fn(batch):

    # Collate doesn't have padding, simply combines everything
    # for later consumption
    x_true, y_true = zip(*batch)
    return torch.stack(x_true), torch.stack(y_true)

# Positional Encoding uses interleaved positional sinusoidal
# encoding
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

"""
Defines an RNN that operates over language to autoregressively
predict the next token from the ones that have come before 
(storing the previous information into a hidden state)
"""
class Transformer(torch.nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

        # Embedding layer
        self.token_embedding = torch.nn.Embedding(num_embeddings=10000, embedding_dim=256, padding_idx=3)

        # Positional Encoding
        self.position_encoding = PositionalEncoding(d_model=256)

        # We treat a transformer encoder layer like a decoder here by using
        # 
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=1024, dropout=0.1, batch_first=True
        )
        self.decoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=3)


        # Passthrough to make a distribution
        self.fc = torch.nn.Linear(in_features=256,out_features=10000)

        # Used during generation
        self.max_len = 512

    # During autoregression, there is no hidden state, just one output
    def forward(self, x):
        batch_size, seq_len = x.size()
        
        # We add together the token embeddings and positional encoding,
        # to maintain information about the relative positions of tokens
        embed = self.token_embedding(x)

        embed = self.position_encoding(embed)

        # We make the attention at the positions we can't see negative
        # infinity, so they do not affect the attention score
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)

        # Finally, pass the target and mask to our encoder
        out = self.decoder(
            src=embed,
            mask=causal_mask
        )

        # Pass decoder output through to fully connected layer
        logits = self.fc(out)

        return logits
    
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
                y_pred = model(x_true)

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
    
    @torch.no_grad()
    def prompt(self, prompt, sp_model, max_new_tokens=20, device="cuda"):
        self.eval()

        prompt_ids = sp_model.encode(prompt, out_type=int)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)

        for _ in range(max_new_tokens):

            if input_tensor.size(1) > self.max_len:
                input_tensor = input_tensor[:, :self.max_len:]

            logits = self.forward(input_tensor)

            # Flatten distribution output provided by RNN (shows the likely next tokens)
            next_token_logits = logits[:, -1, :] / 0.8 # temperature is 1.5

            # Run this distribution through softmax and sample for the next
            # token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Convert token to 1D tensor
            next_token = torch.tensor([[next_token]], dtype=torch.long, device=device)

            # At each generation stage, the new token is added to the end
            # of the input tensor, and the whole thing is run through the
            # model again
            input_tensor = torch.cat([input_tensor, next_token], dim=1)

            # If the next token ID is eos, then break
            if(next_token.item()) == 2:
                break
        
        return sp_model.decode(input_tensor[0].tolist())

    def train_model(self, model, train_loader, validation_loader, optimizer=None, start_epoch=0, epochs=5, lr=1e-3):

        # We define -100 to be the ignore index, as this integer
        # is used to denote masked tokens in the target 
        loss_func = torch.nn.CrossEntropyLoss()

        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2)

        for epoch in range(start_epoch, epochs):

            # Switch model back to training mode after each eval
            # after the end of each epoch
            model.train()

            training_loss = 0

            for x_true, y_true in train_loader:

                # Batch data is in same device as model
                x_true = x_true.to(next(model.parameters()).device)
                y_true = y_true.to(next(model.parameters()).device)

                optimizer.zero_grad()

                # Predict y output from dataloader
                # (During training we throw out the hidden state)
                y_pred = model(x_true)

                loss = loss_func(
                    # We need to flatten the output of the predicted
                    # classes of the predicted output against the
                    # true output
                    y_pred.view(-1, y_pred.size(-1)),
                    y_true.view(-1)
                )

                # print("Training Loss : ",loss)

                # Backpropagate Loss, to update parameters in the model
                loss.backward()

                # Gradient clipping is done to stabilize training in RNNs
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Advance optimizer according to learning rate
                optimizer.step()

                # This is kept for record keeping and printing later
                training_loss += loss

                print("Loss for this batch : ",loss)
            
            # Run evaluation
            validation_loss = self.evaluate_model(model=model, validation_loader=validation_loader, loss_func=loss_func)

            # Advance learning rate scheduler (decrease on plateau), using the validation loss.
            scheduler.step(validation_loss / len(validation_loader))

            # Save checkpoint
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'validation_loss': validation_loss/len(validation_loader),
                'training_loss' : training_loss/len(train_loader),
            }, f"checkpoints/checkpoint_epoch_{epoch+1}_{validation_loss/len(validation_loader)}.pt")

            print(f"Epoch {epoch+1}, Training Loss: {current_loss/len(train_loader):.4f}, Validation Loss: {validation_loss / len(validation_loader):.4f}")
            print(model.prompt(prompt="Which do you prefer? Dogs or cats? ",sp_model=sp))

import tqdm

def perplexity(model, test_loader):

    total_loss = 0
    total_tokens = 0

    # We get the cross entropy loss over the validation set
    for sample_num, (x, y) in tqdm.tqdm(enumerate(test_loader), desc="Calculating Perplexity Score", total=len(test_loader), unit="samples"):

        # if(loop == 0):
        #     print(x[0])
        #     print(y[0])
        #     sp_model.decode(x[0].tolist())
        #     sp_model.decode(y[0].tolist())

        # Send x and y to the cuda device (GPU)
        x = x.to("cuda")
        y = y.to("cuda")

        # Predict y token scores at each position
        # in each sequence
        logits = model(x)

        loss_function = torch.nn.CrossEntropyLoss(ignore_index=3)

        # flatten logits into a 2D Tensor, where each batch is lined
        # up on their length and each token in these is listed as a
        # score distribution
        #
        # This essentially melts logits into (batch * seq_len) X (vocab_size)
        # (logits.size(-1) is vocab_size, -1 is inferred)
        flattened_logits = logits.view(-1, logits.size(-1))

        # Purely flatten y into a 1D tensor
        flattened_y = y.view(-1)

        # Compare logits scores to ground truth
        #
        # Note this also computes NLL, which is then reversed during
        # exponentiation
        loss = loss_function(flattened_logits, flattened_y)

        # Get total number of tokens loss is calculated for
        valid_tokens = (y != 3).sum().item() if (y == 3).any() else y.numel()

        # Get total loss for all valid tokens by multiplying num tokens
        # by average loss
        total_loss += loss.item() * valid_tokens

        # Keep track of total number of tokens loss was calculated for
        total_tokens += valid_tokens



        # if(loop == 0):
        #     print(logits)

        # if(loop == 0):
        #     print("Logits (perplexity) : ",logits)
        #     print("Logits shape : ", logits[0], logits[1])

        # loop += 1
    
    # Return the loss figure over the total number of tokens (essentially
    # e^(average loss))
    return torch.exp(torch.tensor(total_loss / total_tokens))

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def bleu(model, test_data, sp_model):

    smoothing_function = SmoothingFunction().method1

    scores = []

    for idx, item in tqdm.tqdm(enumerate(test_data), desc="Calculating BLEU Score", total=len(test_data), unit="samples"):

        prompt = item["prompt"]
        completion = item["completion"]

        # This doesn't really occur in the test example, but
        # it could potentially cause an out of bounds issue
        if(len(prompt) < 5):
            continue

        total = sp_model.encode(prompt + " " + completion, out_type = int)

        # Just shift the completion to contain a couple more tokens
        prompt = total[:len(prompt)-2]
        completion = total[-1 - len(completion) - 1:]

        prompt_str = sp_model.decode(prompt)

        reference = sp_model.decode(completion)
        reference_tokens = sp_model.encode(reference, out_type=str)

        generated = model.prompt(prompt_str, sp_model)

        candidate = sp_model.encode(generated, out_type=str)

        # See how much prompt and completion align essentially
        score = sentence_bleu([reference_tokens], candidate, smoothing_function=smoothing_function)

        scores.append(score)
    
    # Get average BLEU score over the entire test set essentially
    return sum(scores) / len(scores) if scores else 0.0

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
    train_dataset = TransformerDataset(data=train_data, sp_model=sp, block_size=512)
    validation_dataset = TransformerDataset(data=validation_data, sp_model=sp, block_size=512)

    print(type(train_dataset))
    print(type(validation_dataset))


    # Convert datasets into data loaders to be consumed by model
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=128, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=128, 
        shuffle=False, # No need to shuffle validation data
        collate_fn=collate_fn
    )


    # Set up for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    start_epoch = 0

    TRAIN = False

    USE_CHECKPOINT = True

    if(USE_CHECKPOINT):
        path = "model.pt"

        start_epoch, validation_loss = load_checkpoint(model, optimizer, path)

        print(f"Loading old checkpoint with Validation Loss {validation_loss:.4f}.")

    if(TRAIN):
        model.train_model(model, train_loader, validation_loader, optimizer, start_epoch)

    # Manual Evaluation

    print(model.prompt(prompt="Which do you prefer? Dogs or cats? ",sp_model=sp))

    print(model.prompt(prompt="How was your day this morning?", sp_model=sp, max_new_tokens=128))

    # Testing

    # Loading the model requires an enormous amount of GPU memory
    torch.cuda.empty_cache()

    test_data = []

    # Open test dataset after running model
    with open("../data/test.jsonl", "r") as f:
        for line in f:
            test_data.append(json.loads(line))

    # Convert raw data into dataset for perplexity
    test_dataset = TransformerDataset(data=test_data, sp_model=sp)

    # Convert datasets into data loaders to be consumed by model
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=128, 
        shuffle=False, 
        collate_fn=collate_fn,
        pin_memory=True
    )

    print("Perplexity Score : ", perplexity(model, test_loader)) # 171.7867
    print("BLEU Score : ",bleu(model, test_data, sp)) # 

