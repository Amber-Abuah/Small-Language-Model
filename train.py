from tokenise import *
from model import SmallTransformer, init_weights
import torch
import torch.nn as nn
from early_stop import EarlyStopper
from generate import generate_text
from torch.utils.data import TensorDataset, DataLoader

### Setting hyperparameters for training ------------------------------------------------------------
d_model=128
num_heads=4
num_layers=4
max_len=INPUT_LENGTH
epochs = 100
batch_size = 256
early_stop_patience = 5
scheduler_patience = 3
early_stop_delta = 0.001

### Creating model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallTransformer(vocab_size, d_model, max_len, num_heads, num_layers).to(device)
model.apply(init_weights)
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_index, reduction='sum', label_smoothing=0.1)

print("Number of model parameters:", sum(p.numel() for p in model.parameters()))

### Training loop ----------------------------------------------------------------------------------

def train(inputs, targets, epochs=epochs, batch_size=batch_size, start_prompt="", min_epochs=0, lr=0.001, output_epochs=10):

    optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=scheduler_patience)
    early_stopper = EarlyStopper(early_stop_patience, early_stop_delta)

    inputs = torch.stack([torch.tensor(i).to(device) for i in inputs])
    targets = torch.stack([torch.tensor(t).to(device) for t in targets])
    dataset = TensorDataset(inputs, targets)
    train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


    for e in range(epochs):
        t_loss = 0
        total_tokens = 0

        model.train()

        for x, y in train_loader:
            predictions = model(x)
            predictions = predictions.view(-1, vocab_size)
            y = y.view(-1)
            loss_val = loss_fn(predictions, y)

            optimiser.zero_grad()
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            t_loss += loss_val.item()
            total_tokens += (y != pad_token_index).sum().item()

        train_loss = t_loss/total_tokens
        val_loss = validation(val_loader)
        early_stopper.append_val_loss(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch {e}, Training Loss:{train_loss}.  Validation Loss: {val_loss}")

        if e % output_epochs == 0 and e != 0:
            print(generate_text(model, tokeniser, device, pad_token_index, max_len))

        if early_stopper.should_stop_early() and e > min_epochs:
            print("Early stopping applied.")
            break

def validation(val_loader):
    model.eval()
    t_loss = 0
    t_tokens = 0

    with torch.no_grad():
        for x, y in val_loader:
            predictions = model(x)
            predictions = predictions.view(-1, vocab_size)
            y = y.view(-1)
            t_loss += loss_fn(predictions, y).item()
            t_tokens += (y != pad_token_index).sum().item()

    return t_loss/t_tokens


### Training on all datasets ------------------------------------------------------------
TINY_EPOCHS = 150
VN_EPOCHS = 75
DOKI_EPOCHS = 10

train(tiny_inputs, tiny_targets, TINY_EPOCHS, 128, min_epochs=50)
print("Finished tiny training.")
print(generate_text(model, tokeniser, device, pad_token_index, max_len))

model.freeze_layers(2)
train(vn_inputs, vn_targets, VN_EPOCHS, 128, min_epochs=20, output_epochs=5)
print("Finished VN training.")
print(generate_text(model, tokeniser, device, pad_token_index, max_len))

model.freeze_layers(3)
doki_start_prompt = "<USER>: \""
train(doki_inputs, doki_targets, DOKI_EPOCHS, 32, doki_start_prompt, min_epochs=10, output_epochs=5)
print("Finished Doki Doki training.")

torch.save(model.state_dict(), "doki_slm.pth")

for i in range(3):
    print(generate_text(model, tokeniser, device, pad_token_index, max_len, doki_start_prompt) + "\n")