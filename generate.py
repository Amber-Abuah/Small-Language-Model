import torch
from tokenise import *

# Parse output from model after training
def parse_output(output):
    output = output.replace(newline_token, "\n")
    for s in special_tokens:
        output = output.replace(s, "")
    return output.strip()

# Generate output from full/ partially trained model
def generate_text(model, tokeniser, device, pad_token_index, block_size, start_prompt="", generate_len = 150):
    temp = 0.6
    top_k = 20

    model.eval()
    start_tokens = tokeniser.encode(start_prompt).ids
    tokens = [pad_token_index] * (block_size - len(start_tokens)) + start_tokens

    token_to_count = {}

    for t in start_tokens:
        if t not in token_to_count:
            token_to_count[t] = 0

        token_to_count[t] += 1

    with torch.no_grad():
        for _ in range(generate_len):
            x = torch.tensor(tokens[-block_size:]).to(device).unsqueeze(0)
            logits = model(x)[0, -1] / temp
            probs = torch.softmax(logits, dim=0)

            # Apply penalty to already generated tokens to prevent repetitive outputs
            repeated_ngram_tokens = repeated_ngram_penalty(tokens)

            for k, v in token_to_count.items():
                probs[k] /= (1.0 + 0.9 * v) + (2 * int(k in repeated_ngram_tokens))

            probs[tokens[-1]] = 0
            probs = probs / probs.sum()

            topk_probs, topk_indices = probs.topk(top_k)
            next_token = topk_indices[torch.multinomial(topk_probs, 1)].item()

            if next_token not in token_to_count:
                token_to_count[next_token] = 0
            token_to_count[next_token] += 1

            tokens.append(next_token)

    return parse_output(tokeniser.decode(tokens, skip_special_tokens=False))

# Apply penalty for repeated trigrams
def repeated_ngram_penalty(tokens, n=3):
    ngram_to_freq = {}
    split_str = "<SPLIT>"

    for i in range(len(tokens) - n):
        ngram = tokens[i:i+n]
        ngram_str = split_str.join([str(n) for n in ngram])

        if ngram_str not in ngram_to_freq:
            ngram_to_freq[ngram_str] = 0
        ngram_to_freq[ngram_str] += 1

    repeated_tokens = []

    for k, _ in ngram_to_freq.items():
        repeated_tokens.extend(k.split(split_str))

    return list(set([int(r) for r in repeated_tokens]))
