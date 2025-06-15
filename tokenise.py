from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import tempfile 
import re

INPUT_LENGTH = 150
MAX_VOCAB_SIZE = 6000
train_text_dir = "training_text"

### Text preprocessing functions -----------------------------------------------------------------

# Remove non-relevant dialogue lines
def preprocess_text(s):
    s = re.sub(r"{[^}]*}", "", s)
    s = re.sub(r"\[[^\]]*\]", "", s)
    s = re.sub(r"\*[^\*]*\.png\*", "", s)
    s = re.sub(r"={80}", "", s)
    return "".join(_s for _s in s if _s.isascii())

# Remove common lines with no semantic meaning
def remove_common_dialogue(lines):
    common_lines = ["Ah...", "Sorry...", "I'm sorry...", "..."]
    all_lines = []

    for l in lines:
        if '"' in l:
            if l.split('"')[1] in common_lines:
                continue
        all_lines.append(l)

    return all_lines

### Tokeniser creation -----------------------------------------------------------------
newline_token = "<N>"
unknown_token = "<UNK>"
special_tokens = ["<PAD>", unknown_token, newline_token]

tokeniser = Tokenizer(BPE(unk_token=unknown_token))

tokeniser.normalizer = normalizers.Sequence([
    normalizers.NFD(),
    normalizers.Lowercase(),
    normalizers.StripAccents()
])
tokeniser.pre_tokenizer = Whitespace()

### Gathering data for training -------------------------------------------------------
TINY_MAX_LINES = 40000
lines = open(train_text_dir + "/tiny_stories.txt", errors="replace").readlines()
tiny_full_text = preprocess_text(newline_token.join(lines[:TINY_MAX_LINES]))
full_vocab = tiny_full_text

VN_MAX_LINES = 150000
lines = open(train_text_dir + "/visual_novels.txt", errors="replace").readlines()
lines = remove_common_dialogue(lines)
vn_full_text = preprocess_text(newline_token.join(lines[:VN_MAX_LINES]))
full_vocab += vn_full_text

lines = open(train_text_dir + "/doki.txt", errors="replace").readlines() + open(train_text_dir + "/doki_synthetic.txt", errors="replace").readlines()
lines = remove_common_dialogue(lines)
doki_full_text = preprocess_text(newline_token.join(lines[:2293] + lines[3404:])) # Remove consecutive Monika lines
full_vocab += doki_full_text

### Tokeniser training -----------------------------------------------
with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as tmp:
    tmp.write(full_vocab)
    trainer = BpeTrainer(vocab_size = MAX_VOCAB_SIZE)
    tokeniser.train([tmp.name], trainer)

tokeniser.add_special_tokens(special_tokens)

### Creating input and target data/ tokens for training -----------------------------------------------

def tokenise_text(text, sliding_window=1):
    encoded_text = tokeniser.encode(text)
    tokens = encoded_text.ids

    inputs = []
    targets = []

    for i in range(0, len(tokens) - INPUT_LENGTH, INPUT_LENGTH // sliding_window):
        inputs.append(tokens[i:i+INPUT_LENGTH])
        targets.append(tokens[i+1:i+1 + INPUT_LENGTH])

    return inputs, targets


tiny_inputs, tiny_targets = tokenise_text(tiny_full_text)
vn_inputs, vn_targets = tokenise_text(vn_full_text)
doki_inputs, doki_targets = tokenise_text(doki_full_text, 4)

pad_token_index = tokeniser.encode(special_tokens[0]).ids[0]
vocab_size = tokeniser.get_vocab_size()
