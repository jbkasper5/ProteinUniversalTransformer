import torch
import torch.nn as nn
import pytorch_lightning as pl

# Notes:
# nn.Linear takes in 3 main arguments: in_features, out_features, and bias
# in_features is the size of the input sample, out_features is the size of the output sample, and bias determines if we have addidtive bias
# nn.Linear computes y (out_features) = (x (in_features) • A.transpose) + bias, where A is a matrix defined by the linear layer
# in other words, an nn.Linear object defines a matrix such that the input shape (*, in) combined with the matrix produces a matrix of shape (*, out)
#       thus, nn.Linear -> (*, in) • (in, out) = (*, out)
#       thus, nn.Linear -> x • A.transpose = (*, out)
#       thus, (in, out) = A.transpose, thus A = (out, in)
#       nn.Linear creates a matrix that is of shape (out_features, in_features)

# defines the self attention module for the transformer, which is to be passed into the feed forward layers
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        # initialize the pl.LightningModule class, tells python that this class is a neural net class
        super(SelfAttention, self).__init__()

        # set embed size locally, which becomes the shape of the output of the self attention layer (and the shape of the input of the encoder stack)
        self.embed_size = embed_size

        # set the number of attention heads locally
        self.heads = heads

        # calculate the shape of the vectors, since the end product will be an [head] concatenated matrix that must be [embed_size] big
        self.head_dim = embed_size // heads

        # assert that the embed size is evenly divisible by the number of heads
        # since we need to concatenate all of the attention head output matrices, we have h head matrices, meaning we need to match the dimensions of the matrix
        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by number of attention heads"

        # values becomes a matrix that is of shape (self.head_dim, self.head_dim), with no bias
        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)

        # keys becomes a matrix that is of shape (self.head_dim, self.head_dim), with no bias
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)

        # queries becomes a matrix that is of shape (self.head_dim, self.head_dim), with no bias
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False)

        # fc_out is the output of a fully connected layer, and is of shape (embed_size, (heads * self.head_dim)), bias is true
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # N is equal to our batch size
        N = query.shape[0]

        # since the first index is the batch size, the length of each tensor is index 1 of the shape tuple
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # reshape the keys, values, and queries to account for attention heads
        # the new shape will be (batch_size, [len], 8, 64) in the default case
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # compute all of the key, value, and query vectors by performing the matrix multiplication with the Key, Value, and Query vectors
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum('nqhd,nkhd->nhqk', [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim)     (N, [len], 8, 64)
        # key shape: (N, key_len, heads, heads_dim)           (N, [len], 8, 64)
        # energy shape: (N, 8, query_len, key_len)            (N, 8, [len], [len])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        # compute the usual softmax operation before multiplying with the values and reshaping 
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim = 3)

        # we first compute the matrix product of our softmax and the value pairs, then reshape it to match the embed_size
        # the reshaping is the concatenation step described in the post
        out = torch.einsum('nhql,nlhd->nqhd', [attention, values]).reshape(N, query_len, (self.heads * self.head_dim))
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, head_dim)
        # want: (N, query_len, heads, head_dim)

        # fc_out defines a matrix that is of size (embed_size, embed_size)
        # the out matrix going in is of size ([len], embed_size)
        # thus, the final output of self attention is ([len], embed_size) • (embed_size, embed_size) = ([len], embed_size)
        # this matches the initial input that was supposed to be fed into the model
        out = self.fc_out(out)
        return out # of shape ([len], embed_size)

# defines one encoder block in the encoder stack
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        # initialize the pl.LightningModule class to add this class as a model module
        super(TransformerBlock, self).__init__()

        # compute self attention from the embed_size (512) and the number of heads (8)
        self.attention = SelfAttention(embed_size, heads)

        # define the first layer norm, taking the embedding size as input (512)
        self.norm1 = nn.LayerNorm(embed_size)

        # define the second layer norm, taking the embedding size as input (512)
        self.norm2 = nn.LayerNorm(embed_size)

        # define the feed_forward structure used in the encoder block
        # this feed_forward model consists of two linear units and a relu activation function
        # our input must be (x, 512), then gets matrix multiplied by (512, 2048) and (2048, 512)
        # thus, the input gets multiplied to produce ([len], 2048) • (2048, 512) producing the original ([len], 512) matrix
        self.feed_forward = nn.Sequential(
            # in_features = embed_size, out_features = forward_expansion (4) * embed_size
            # matrix A = (4 * 512, 512) = (2048, 512)
            nn.Linear(embed_size, (forward_expansion * embed_size)),
            #pass through relu
            nn.ReLU(),
            # second linear feature has a matrix inverse that of the first, specifically (512, 2048)
            nn.Linear((forward_expansion * embed_size), embed_size)
        )

        # compute dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # NORM THEN ADD VS ADD THEN NORM

        # first pass the values, keys, and queries through self attention with the desired data masking
        attention = self.attention(value, key, query, mask)

        # next compute layer norm by adding the data to attention and taking layernorm, then apply dropout
        x = self.dropout(self.norm1(attention + query))

        # pass the data through the feedforward model
        forward = self.feed_forward(x)

        # compute the second layernorm operation with the output of the feedforward layer and the result of the previous layernorm operation
        out = self.dropout(self.norm2(forward + x))

        # return the result aftre applying the dropout
        return out

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()

        # set the embedding size that characterizes the size of the FNN model in the encoder block
        self.embed_size = embed_size

        # configure for cuda devices
        # self.device = device

        # create word embedding for the source string
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)

        # create positional embedding for the source string
        self.positional_embedding = nn.Embedding(max_length, embed_size)

        # define the layers of the encoder with num_layers encoder blocks
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size = embed_size, heads = heads, dropout = dropout, forward_expansion = forward_expansion) for _ in range(num_layers)
        ])

        # apply dropout when necessary
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # get the batch_size (N) and the sequence length of the input data
        N, sequence_length = x.shape

        # create an array from index 0 to index [sequence_length], and expand it to match the batch size
        positions = torch.arange(0, sequence_length).expand(N, sequence_length) #.to(self.device)

        # apply positional embedding to the index tensor and word embedding to the one hot input tensor, then apply dropout
        out = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))

        # for each encoder block, pass in the processed output of the previous encoder layer for the key, query, and value tensors
        for layer in self.layers:
            out = layer(out, out, out, mask)

        # return the output of the encoder stack
        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()

        # encoder-decoder self attention
        self.attention = SelfAttention(embed_size, heads)

        # standard layer normalization
        self.norm = nn.LayerNorm(embed_size)

        # encoder block
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)

        # standard dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        # note that embed_size is of size 512
        # compute multihead attention
        attention = self.attention(x, x, x, trg_mask) # attention is of shape (batch, [len], embed_size)

        # compute the query matrix from standard layer norm operations
        query = self.dropout(self.norm(attention + x)) # query is of shape (batch, [len], embed_size)

        # pass the keys, queries, and values into the transformer block (since encoders and decoders use identical blocks)
        out = self.transformer_block(value, key, query, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        # self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length) #.to(self.device)
        x = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        out = self.fc_out(x) # the output of the decoder is logits - that is, it hasn't been passed through an activation function
        return out

class Transformer(pl.LightningModule):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size = 512, num_layers = 6, forward_expansion = 4, heads = 8, dropout = 0, device = 'cpu', max_length = 100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size = src_vocab_size, embed_size = embed_size, num_layers = num_layers, heads = heads, device = device, forward_expansion = forward_expansion, dropout = dropout, max_length = max_length)
        self.decoder = Decoder(trg_vocab_size = trg_vocab_size, embed_size = embed_size, num_layers = num_layers, heads = heads, device = device, forward_expansion = forward_expansion, dropout = dropout, max_length = max_length)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        # self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # shape = (N, 1, 1, src_len)
        return src_mask #.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(N, 1, trg_len, trg_len)
        return trg_mask #.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src) # shape = [35, 1, 1, 20]
        trg_mask = self.make_trg_mask(trg) # shape = [35, 1, 20, 20]
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

    def train_step(self, batch, batch_idx):
        batch = self.DataLoader.get_batch(index = batch_idx, split = 'train')
        out = self(batch)
        return #loss(out)

    def validation_step(self, batch, batch_idx):
        batch = self.DataLoader.get_batch(index = batch_idx, split = 'val_90')
        out = self(batch)
        return #loss(out)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
        
# Layout
# information gets embedded
# embedded information passes into the encoder:
#       first moves through self-attention
#       next through layer norm
#       next through feed-forward
#       lastly through another layer norm