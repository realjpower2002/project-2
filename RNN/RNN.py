import torch

# We define the dataset class here
class RNNDataset(torch.utils.data.Dataset):

    # The max prompt and completion lengths are not super long (usually much
    # less than 50 tokens)
    def __init__(self, data, sp_model, max_total_length = 50):
        self.data = data
        self.sp = sp_model
        # self.max_prompt_length = max_prompt_length
        # self.max_completion_length = max_completion_length
        self.max_total_length = max_total_length

    # The length of the dataset is defined just by the number of
    # lines in it
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # We encode the prompt and completion into token ids
        # on-demand as batches are created
        # prompt_ids = self.sp.encode(item["prompt"], out_type=int)[-self.max_prompt_length:]
        
        # completion_ids = self.sp.encode(item["completion"], out_type=int)[:self.max_completion_length]

        full_text = item["prompt"] + " " + item["completion"]

        tokens = self.sp.encode(full_text, out_type=int)[:self.max_total_length+1]

        # Just in case there is a very small prompt/completion pair, we
        # just move on to the next item
        if len(tokens) < 2:
            # Not enough to form x/y
            return self.__getitem__((idx + 1) % len(self.data))


        # The prompt/completion pairs are actually combined, and each token
        # is simply used to predict the next by shifting the sequence like this

        # Input IDs contains everything except the last token
        input_ids = torch.tensor(tokens[:-1])

        # Target IDs masks everything but the completion part
        target_ids = torch.tensor(tokens[1:])

        return input_ids, target_ids


# Function used for DataLoader class 
def collate_fn(batch):

    # Convert the batch into two lists of input and target sequences
    input_seqs, target_seqs = zip(*batch)

    # Pad the input using the pad value (integer 3)
    input_padded = torch.nn.utils.rnn.pad_sequence(input_seqs, batch_first=True, padding_value = 3)
    
    # Target is padded using mask value (integer -100)
    target_padded = torch.nn.utils.rnn.pad_sequence(target_seqs, batch_first=True, padding_value = 3)

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
        self.embedding = torch.nn.Embedding(num_embeddings=10000, embedding_dim=500, padding_idx=3)

        # Recurrent layers
        self.rnn = torch.nn.RNN(input_size=500, hidden_size=500, num_layers=6, batch_first=True)

        # Passthrough to make a distribution
        self.fc = torch.nn.Linear(in_features=500,out_features=10000)

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
    
    @torch.no_grad()
    def prompt(self, prompt, sp_model, max_new_tokens=20, device="cuda"):
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
            #
            # This needs to be cast to float because cuda can cause precision issues
            # sometimes.
            next_token_output = (output[:, -1, :]).float() / 0.8 # temperature is 0.8

            # Run this distribution through softmax and sample for the next
            # token
            probs = torch.softmax(next_token_output, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token_id)

            # print(next_token_id, sp_model.decode([next_token_id]))

            # If the next token ID is eos, then break
            if(next_token_id) == 2:
                break
        
        return sp_model.decode(generated)
    
    def train_model(self, model, train_loader, validation_loader, optimizer=None, start_epoch=0, epochs=300, lr=1e-4):

        # We define -100 to be the ignore index, as this integer
        # is used to denote masked tokens in the target 
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=3)

        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

        for epoch in range(start_epoch, epochs):

            # Switch model back to training mode after each eval
            # after the end of each epoch
            model.train()

            training_loss = 0

            for x_true, y_true in train_loader:

                # Batch data is in same device as model
                x_true = x_true.to(next(model.parameters()).device)
                y_true = y_true.to(next(model.parameters()).device)

                # for name, param in model.named_parameters():
                #     if param.grad is not None and torch.isnan(param.grad).any():
                #         print(f"NaN detected in gradients of {name}")


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

                # print("Training Loss : ",loss)

                # Backpropagate Loss, to update parameters in the model
                loss.backward()

                # Gradient clipping is done to stabilize training in RNNs
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Advance optimizer according to learning rate
                optimizer.step()

                # This is kept for record keeping and printing later
                training_loss += loss
            
            # Run evaluation
            validation_loss = self.evaluate_model(model=model, validation_loader=validation_loader, loss_func=loss_func)

            torch.cuda.empty_cache()

            # Advance learning rate scheduler (decrease on plateau), using the validation loss.
            # scheduler.step(validation_loss)

            # Save checkpoint
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'validation_loss': validation_loss/len(validation_loader),
                'training_loss' : training_loss/len(train_loader),
            }, f"checkpoints/checkpoint_epoch_{epoch+1}_{validation_loss/len(validation_loader)}.pt")

            print(f"Epoch {epoch+1}, Training Loss: {training_loss/len(train_loader):.4f}, Validation Loss: {validation_loss / len(validation_loader):.4f}")
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
        logits, _ = model(x)

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
    train_dataset = RNNDataset(data=train_data, sp_model=sp)
    validation_dataset = RNNDataset(data=validation_data, sp_model=sp)


    # Convert datasets into data loaders to be consumed by model
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=512, 
        shuffle=True, 
        collate_fn=collate_fn,
        pin_memory=True
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=512, 
        shuffle=False, # No need to shuffle validation data
        collate_fn=collate_fn,
        pin_memory=True
    )


    # Set up for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = RNN()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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

    print(model.prompt(prompt="How was your day this morning?", sp_model=sp))
    
    # Testing

    test_data = []

    # Convert raw data into dataset for perplexity
    test_dataset = RNNDataset(data=test_data, sp_model=sp)

    # Convert datasets into data loaders to be consumed by model
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=512, 
        shuffle=False, 
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Open test dataset after running model
    with open("../data/test.jsonl", "r") as f:
        for line in f:
            test_data.append(json.loads(line))

    print("Perplexity Score : ", perplexity(model, test_loader)) # Perplexity : 266.0346
    print("BLEU Score : ",bleu(model, test_data, sp)) # BLEU Score :  0.061525491085695175

