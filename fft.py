import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

dropout = 0.1


torch.manual_seed(1337)


def tril_init(linear):
    with torch.no_grad():
        linear.weight.copy_(torch.tril(linear.weight))

# Zero out gradients
def get_zero_grad_hook(mask):
    def hook(grad):
        return grad * mask
    return hook




class Block(nn.Module):
    def __init__(self, time_intervals, n_embed, n_head, tri_W):
        super().__init__()

        self.ffts = nn.Sequential(
            Rearrange("B T (h E) -> B h E T", h = n_head),
            nn.Linear(time_intervals, time_intervals, bias=None),
            Rearrange("B h E T -> B T (h E)", h = n_head)
        )

        self.ffw = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.LeakyReLU(0.07),
            nn.Linear(n_embed, n_embed),
            nn.Dropout(dropout)
        )

        self.ffts[1].apply(tril_init)
        self.ffts[1].weight.register_hook(get_zero_grad_hook(tri_W))

        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, input):
        x = self.ln1(input)
        x = x + self.ffts(x)
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

        tri = torch.tril(torch.ones((time_intervals, time_intervals), dtype=torch.float32)).to(device)
        tri_W = tri/tri.sum(dim=1, keepdim=True)

        self.blocks = nn.Sequential(*[Block(time_intervals, n_embed, n_head, tri_W.detach()) for _ in range(n_layers)])

        
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
