import torch
import torch.nn as nn
import torch.nn.functional as F
import random

dropout = 0.1


class Spike(nn.Module):
    def __init__(self, dim):
        super(Spike, self).__init__()
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):
        x = self.ln(x)
        return (x>0.0)*x*torch.tanh(x)



class FeedForward(nn.Module):

    def __init__(self, f_in, f_out):
        super().__init__()


        self.net = nn.Sequential(
            nn.Linear(f_in, f_in),
            nn.ReLU(),
            nn.Linear(f_in, f_out),
            nn.Dropout(dropout)
        )

    def forward(self, input):
        return self.net(input)


class FourierTransform(nn.Module):
    def __init__(self, device, time_intervals, f_in, f_out):
        super().__init__()
        self.value = nn.Linear(f_in, f_in, bias=False)
        self.ln1 = nn.LayerNorm(f_in)
        self.fft = nn.Linear(time_intervals, time_intervals, bias=None)
        self.ln2 = nn.LayerNorm(f_in)
        self.project = nn.Linear(f_in, f_out, bias=False)
        self.tril = torch.tril(torch.ones((time_intervals, time_intervals))).to(device)
        self.tril_W = self.tril/self.tril.sum(dim=1, keepdim=True)
        
    
    def forward(self, x):
        B,T,E = x.shape
        x = self.ln1(self.value(x))
        x = x.reshape(B, E, T)
        x = F.linear(x, self.tril_W[:T,:T] * 3.0*torch.tanh(self.fft.weight[:T,:T]/3), None)
        x = x.reshape(B, T, E)
        x = self.project(self.ln2(x))
        return x



class Block(nn.Module):
    def __init__(self, device, time_intervals, n_embed, n_head):
        super().__init__()
        head_size = n_embed//n_head
        self.heads = nn.ModuleList([FourierTransform(device, time_intervals, n_embed, head_size) for i in range(n_head)])
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
    def __init__(self, vocab_size, time_intervals, vocab_embed, n_embed, n_head, n_layers, device="cpu"):
        super().__init__()
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_embed)
        self.position_embedding_table = nn.Embedding(time_intervals, vocab_embed)

        self.ln_in = nn.LayerNorm(vocab_embed)
        self.uniform = nn.Linear(vocab_embed, n_embed)

        self.blocks = nn.Sequential(*[Block(device, time_intervals, n_embed, n_head) for _ in range(n_layers)])

        
        self.ln_out = nn.LayerNorm(n_embed)
        
        self.linear_head = nn.Linear(n_embed, vocab_size)

        self.time_intervals = time_intervals

        self.cdist = torch.distributions.categorical



    def forward(self, idx, targets=None):
        
        B, T = idx.shape

        
            
        tok_emb = self.token_embedding_table(idx) # B, T, E
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))

        x = tok_emb + pos_emb

        x = self.uniform(self.ln_in(x))

        embed  = self.ln_out(self.blocks(x))
        logits = self.linear_head(embed)
        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            logits = logits.view(B*T, V)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return embed, logits, loss
    
    
    def decode(self, idx):
        with torch.no_grad():
            _, logits, _ = self(idx)
            probs = F.softmax(logits, dim=-1)
            m = self.cdist.Categorical(probs)
            idx = m.sample()
            return idx

    def generate(self, idx, max_new_tokens, LLM=None):
        #idx is (B, T) array of indices in the current context
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # crop idx to the last block_size tokens
                idx_cond = idx[:, -self.time_intervals:]
                # get the predictions
                idx_cond_next = LLM.decode(idx_cond) if LLM != None else idx_cond
                _, logits, _ = self(idx_cond_next)
                #focus only on the last time step
                logits = logits[:, -1, :] #become (B, C)
                # apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1) #(B, C)
                # sample from distribution
                idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
                # append sampled index to the running sequence
                idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


torch.manual_seed(1337)

batch_size = 64
time_intervals = 200
max_iter = 1000000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 30



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters:", len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])



data = torch.tensor(encode(text), dtype=torch.long)


torch.manual_seed(135665)


def get_batch():
    #var_time = random.randint(32, time_intervals)
    var_time = time_intervals
    ix = torch.randint(len(data) - var_time, (batch_size, ))
    x = torch.stack([data[i:i+var_time] for i in ix])
    y = torch.stack([data[i+1:i+var_time+1] for i in ix])
    return x.to(device), y.to(device)


def get_random_block():
    i = random.randint(0, len(data) - time_intervals)
    block = data[i:i+time_intervals].reshape(1, -1).to(device)
    return block

@torch.no_grad()
def estimate_loss():
    LLM.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch()
        _, _, loss = LLM(X, targets=Y)
        losses[k] = loss.item()
    out = losses.mean()
    LLM.train()
    return out



LLM = BigramLanguageModel(vocab_size, time_intervals, vocab_embed=196, n_embed=196, n_head=14, n_layers=14, device=device).to(device)
optimizer = torch.optim.Adam(LLM.parameters(), lr=learning_rate)

pytorch_total_params = sum(p.numel() for p in LLM.parameters())
print('LLM parameters: ', pytorch_total_params)


try:
    LLM.load_state_dict(torch.load('LLM_model.pt'))
    context = get_random_block()
    print(decode(LLM.generate(context, max_new_tokens=250)[0].tolist())[-250:])
    print("loaded")
except:
    print("no LLM")

for iter in range(max_iter):

    #step = 0

    if iter % eval_interval == 0:
        #nanoFFT.LEARN_BY_HEART = False if step%2==0 else True
        #step += 1
        losses = estimate_loss()
        context = get_random_block()
        text = decode(LLM.generate(context, max_new_tokens=50)[0].tolist())[-50:]
        text = text.replace("\n", " <new line> ")
        print(f"step {iter}, train loss: {losses:.4f}, text: {text}")
        if iter>=1000:
            try:
                torch.save(LLM.state_dict(), 'LLM_model.pt')
            except:
                print("problem during saving LLM")

        if iter>=10000 and iter%10000==0:
            context = get_random_block()
            print(decode(context[0].tolist()))
            print("###########################################")
            print("###########################################")
            print(decode(LLM.generate(context, max_new_tokens=500)[0].tolist()))
            print("###########################################")
            print("###########################################")
    #sample batch of data
    xb, yb = get_batch()

    #evaluate the loss
    _, _, loss = LLM(xb, targets=yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate from the LLM
#context = torch.ones((1,1), dtype=torch.long, device=device)

context = get_random_block()

print(decode(context[0].tolist()))

print("###########################################")
print("###########################################")
print("###########################################")



print(decode(LLM.generate(context, max_new_tokens=500)[0].tolist()))

