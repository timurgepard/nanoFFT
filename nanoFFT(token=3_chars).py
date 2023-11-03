import torch
import torch.nn as nn
import torch.nn.functional as F
import random




batch_size = 64
time_intervals = 256
n_embed = 210
n_head = 30
dropout = 0.2
n_layers = 10
max_iter = 100000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 30
gamma = 1.0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


text_length = len(text)

text = text + ' ' * (3 - (text_length % 3))  # Make sure the text length is divisible by 3 by adding spaces if needed

print("length of dataset in characters:", len(text))

all_tokens = [text[i:i + 3] for i in range(0, len(text), 3)]


tokens = list(set(all_tokens))
vocab_size = len(tokens)
print('vocab length: ', len(tokens))

stoi = {ch:i for i,ch in enumerate(tokens)}
itos = {i:ch for i,ch in enumerate(tokens)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])






data = torch.tensor(encode(all_tokens), dtype=torch.long)
n = int(0.9*len(data))

train_data = data[:n]
val_data = data[n:]


torch.manual_seed(1337)


def get_batch(split_type):
    data = train_data if split_type == "train" else val_data
    ix = torch.randint(len(data) - time_intervals, (batch_size, ))
    x = torch.stack([data[i:i+time_intervals] for i in ix])
    y = torch.stack([data[i+1:i+time_intervals+1] for i in ix])
    return x.to(device), y.to(device)


def get_random_block():
    data = train_data
    i = random.randint(0, len(data) - time_intervals)
    block = data[i:i+time_intervals].reshape(1, -1).to(device)
    return block

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out




class FeedForward(nn.Module):

    def __init__(self, f_in, f_out):
        super().__init__()


        self.net = nn.Sequential(
            nn.LayerNorm(f_in),
            nn.Linear(f_in, f_in),
            nn.ReLU(),
            nn.Linear(f_in, f_out),
            nn.Dropout(dropout)
        )

    def forward(self, input):
        return self.net(input)


class FourierTransform(nn.Module):
    def __init__(self, time_intervals, f_in, f_out):
        super().__init__()

        self.value = nn.Linear(f_in, f_in, bias=False)
        self.fft = nn.Linear(time_intervals, time_intervals, bias=False)
        self.project = nn.Linear(f_in, f_out, bias=False)

        self.tril = torch.tril(torch.ones((time_intervals, time_intervals))).to(device)
        self.tril_W = self.tril/self.tril.sum(dim=1, keepdim=True)

        
    def forward(self, input):

        B,T,E = input.shape
        x = self.value(input)
        x =x.reshape(B, E, T)
        x = nn.functional.linear(x, self.tril_W[:T,:T] * self.fft.weight[:T,:T], None)
        x = x.reshape(B, T, E)
        return self.project(x)



class Block(nn.Module):
    def __init__(self,  n_head):
        super().__init__()
        head_size = n_embed//n_head
        self.heads = nn.ModuleList([FourierTransform(time_intervals, n_embed, head_size) for i in range(n_head)])
        self.ffw = FeedForward(n_embed, n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, input):
        x = self.ln1(input)
        x = x + torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.ln2(x)
        out = x + self.ffw(x)
        return out




class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(time_intervals, n_embed)

        self.blocks = nn.Sequential(*[Block(n_head) for _ in range(n_layers)])

        self.normalize = nn.LayerNorm(n_embed)
        
        self.linear_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # B, T, C
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))

        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.normalize(x)
        logits = self.linear_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    


    def generate(self, idx, max_new_tokens):
        #idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -time_intervals:]
            # get the predictions
            logits, loss = self(idx_cond)
            #focus only on the last time step
            logits = logits[:, -1, :] #become (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) #(B, C)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx



model = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

pytorch_total_params = sum(p.numel() for p in model.parameters())
print('model parameters: ', pytorch_total_params)

try:
    model.load_state_dict(torch.load('nanoFFT_model.pt'))
    context = get_random_block()
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
except:
    print("no model")

for iter in range(max_iter):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        context = get_random_block()
        text = decode(model.generate(context, max_new_tokens=50)[0].tolist())[-50:]
        text = text.replace("\n", " <new line> ")
        print(f"step {iter}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}, text: {text}")
        if iter>=2000:
            try:
                torch.save(model.state_dict(), 'nanoFFT_model.pt')
            except:
                print("problem during saving model")

        if iter>=10000 and iter%10000==0:
            context = get_random_block()
            print(decode(context[0].tolist()))
            print("###########################################")
            print("###########################################")
            print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
            print("###########################################")
            print("###########################################")
    #sample batch of data
    xb, yb = get_batch("train")

    #evaluate the loss
    logits, loss = model(xb, targets=yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate from the model
#context = torch.ones((1,1), dtype=torch.long, device=device)

context = get_random_block()

print(decode(context[0].tolist()))

print("###########################################")
print("###########################################")
print("###########################################")



print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))


