from protein_model.transformer import Transformer
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

def data_process(raw_text_iter):
    # """Converts raw text into a flat Tensor."""
    data = []
    words = []
    for item in raw_text_iter:
        # print(f'{item} -> {tokenizer(item)} -> {vocab(tokenizer(item))}')
        words.append(tokenizer(item))
        data.append(torch.tensor(vocab(tokenizer(item)), dtype=torch.long))
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data))), [item for item in words if item != []]

# train_iter was "consumed" by the process of building the vocab,
# so we have to create it again
train_iter, val_iter, test_iter = WikiText2()
train_data, train_text = data_process(train_iter)
val_data, val_text = data_process(val_iter)
test_data, test_text = data_process(test_iter)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def batchify(data, bsz: int):
    # """Divides the data into bsz separate sequences, removing extra elements
    # that wouldn't cleanly fit.

    # Args:
    #     data: Tensor, shape [N]
    #     bsz: int, batch size

    # Returns:
    #     Tensor of shape [N // bsz, bsz]
    # """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

# cuts a sequence into [batch_size] equal parts, so essentially it's max_length
batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)


# this is how many examples to steal
bptt = 35
def get_batch(source, i: int):
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    # sequence length is the minimum of bptt and 
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len]#.reshape(-1)
    return data, target


if __name__ == '__main__':
    print('break here!')
    # exit()
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 33278
    trg_vocab_size = 33278
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(train_data.shape)
    data, targets = get_batch(train_data, 0)
    print(data.shape)
    # targets = targets.unsqueeze(0)
    model = Transformer(
                src_vocab_size = src_vocab_size, 
                trg_vocab_size = trg_vocab_size, 
                src_pad_idx = src_pad_idx, 
                trg_pad_idx = trg_pad_idx, 
                dropout = 0.2, 
                embed_size = 200,
                num_layers = 2,
                heads = 2
            ).to(device)

    # SHAPES
        # input to transformer: 
        # input to decoder: 
        # output of self attention: [35, 20, 200]
        # output of ff layer in encoder: [35, 20, 200]
        # output of encoder: [35, 20, 200]
        # input shape to decoder: [35, 20]
        # output of attention in decoder: [35, 20, 200]
        # shape of data after passing through decoder sublayers: [35, 20, 200]
        # shape of data after leaving first encoder block: [35, 20, 33278]
    # model = torch.nn.Transformer(d_model = len(vocab), nhead = 2, dim_feedforward = 50)
    print(targets.shape)
    out = model(data, targets)
    print(out.shape)
    print(out)
