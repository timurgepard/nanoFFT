import torch
import random
from fft import BigramLanguageModel
import pickle

torch.manual_seed(1337)
scaler = torch.cuda.amp.GradScaler()

batch_size = 64
time_intervals = 300
max_iter = 1000000
eval_interval = 500
learning_rate = 3e-5
eval_iters = 10



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


with open('./input/tokens.pkl', 'rb') as f:
    tokens  = pickle.load(f)

with open('./input/input.pkl', 'rb') as f:
    input_tokens  = pickle.load(f)

vocab_size = len(tokens)
print('vocab token size: ', vocab_size)
print('input text token size: ', len(input_tokens))


stoi = {ch:i for i,ch in enumerate(tokens)}
itos = {i:ch for i,ch in enumerate(tokens)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])




data = torch.tensor(encode(input_tokens), dtype=torch.long)


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



LLM = BigramLanguageModel(vocab_size, time_intervals, vocab_embed=800, n_embed=800, n_head=20, n_layers=20, device=device).to(device)
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
    with torch.cuda.amp.autocast(dtype=torch.float16):
        _, _, loss = LLM(xb, targets=yb)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


#generate from the LLM
#context = torch.ones((1,1), dtype=torch.long, device=device)

context = get_random_block()

print(decode(context[0].tolist()))

print("###########################################")
print("###########################################")
print("###########################################")



print(decode(LLM.generate(context, max_new_tokens=500)[0].tolist()))


