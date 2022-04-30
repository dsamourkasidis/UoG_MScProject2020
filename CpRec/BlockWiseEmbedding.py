import numpy as np
import torch
import torch.nn as nn


class BlockWiseEmbeddingForInput(nn.Module):
    """
    # input shape
        2D tensor with shape：[batch_size, seq_len]
    # return
        3D tensor with shape：[batch_size, seq_len, output_dim]
    """

    def __init__(self, vocab_size, embed_dim, device, block=None, block_factor=4):
        super(BlockWiseEmbeddingForInput, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.block = block
        self.block_factor = block_factor

        block_num = len(block) - 1
        # block factor = 1 means simple embedding
        if block_factor == 1:
            stdv = np.sqrt(1. / vocab_size)
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_dim
            )
            nn.init.uniform_(self.embedding.weight, -stdv, stdv)
        else:
            otherblock_dims = []
            curr_block_factor = self.block_factor
            for i in range(block_num):
                dim = max(1, self.embed_dim / curr_block_factor)
                otherblock_dims.append(int(dim))
                curr_block_factor *= self.block_factor

            firstblock_K = self.block[0]

            self.firstblock_w = nn.Embedding(num_embeddings=firstblock_K, embedding_dim=self.embed_dim)
            self.otherblock_w = []
            for i in range(block_num):
                block_i_dim = otherblock_dims[i]
                block_i_K = self.block[i + 1] - self.block[i]
                self.otherblock_w.append([
                    torch.rand([block_i_dim, self.embed_dim], requires_grad=True).to(device),
                    nn.Embedding(num_embeddings=block_i_K, embedding_dim=block_i_dim).to(device),
                ])

    def forward(self, inputs):
        # print("shape: ", inputs.shape)
        # inputs: [batch_size, seq_len]
        if self.block_factor == 1:
            outputs = self.embedding(inputs)
            print('using embeddding')
        else:
            input_size = list(inputs.shape)
            # print("input_size: ", input_size)
            outputs = torch.zeros(input_size + [self.embed_dim], dtype=torch.float32).to(self.device)

            block_value = [0] + self.block
            for i in range(len(block_value) - 1):
                low_idx = block_value[i]
                high_idx = block_value[i + 1]
                mask = torch.logical_and(torch.ge(inputs, low_idx), torch.lt(inputs, high_idx)).int()

                if i == 0:
                    firstblock_inputs = (inputs - low_idx) * mask
                    firstblock_embed = self.firstblock_w(firstblock_inputs)
                    projected = firstblock_embed  # [batch_size, seq_len, output_dim]
                else:
                    block_i_inputs = (inputs - low_idx) * mask
                    block_i_embed = torch.matmul(self.otherblock_w[i - 1][1](block_i_inputs),
                                                 self.otherblock_w[i - 1][0])
                    projected = block_i_embed

                outputs += projected * torch.unsqueeze(mask, dim=-1)
        return outputs
