import pickle


with open('./input/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.encode("ascii", "ignore")
text = text.decode()

text = text + ' ' * (3 - (len(text) % 3))  # Make sure the text length is divisible by 3 by adding spaces if needed


text = text.replace('\n\n', '\n')
text = text.replace('  ', ' ')


text_tokens = [text[i:i + 3] for i in range(0, len(text), 3)]


tokens = list(set(text_tokens))

with open('./input/tokens.pkl', 'wb+') as f:
    pickle.dump(tokens, f)

with open('./input/input.pkl', 'wb+') as f:
    pickle.dump(text_tokens, f)

