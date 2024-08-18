import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
class DenseLayer(nn.Module):

    def __init__(self,
                 c_in,  # dimensionality of input features
                 c_out,  # dimensionality of output features
                 zero_init=False,  # initialize weights as zeros; use Xavier uniform init if zero_init=False
                 ):

        super().__init__()

        self.linear = nn.Linear(c_in, c_out)

        # Initialization
        if zero_init:
            nn.init.zeros_(self.linear.weight.data)
        else:
            nn.init.uniform_(self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
        nn.init.zeros_(self.linear.bias.data)

    def forward(self,
                node_feats,  # input node features
                ):

        node_feats = self.linear(node_feats)

        return node_feats

class GATSingleHead(nn.Module):

    def __init__(self,
                 c_in,  # dimensionality of input features
                 c_out,  # dimensionality of output features
                 temp=1,  # temperature parameter
                 ):
        super().__init__()

        self.linear = nn.Linear(c_in, c_out)
        self.v0 = nn.Parameter(torch.Tensor(c_out, 1))
        self.v1 = nn.Parameter(torch.Tensor(c_out, 1))
        self.temp = temp

        # Initialization
        nn.init.uniform_(self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
        nn.init.zeros_(self.linear.bias.data)
        nn.init.uniform_(self.v0.data, -np.sqrt(6 / (c_out + 1)), np.sqrt(6 / (c_out + 1)))
        nn.init.uniform_(self.v1.data, -np.sqrt(6 / (c_out + 1)), np.sqrt(6 / (c_out + 1)))

    def forward(self,
                node_feats,  # input node features
                adj_matrix,  # adjacency matrix including self-connections
                ):
        # Apply linear layer and sort nodes by head
        node_feats = self.linear(node_feats)
        f1 = torch.matmul(node_feats, self.v0)
        f2 = torch.matmul(node_feats, self.v1)
        attn_logits = adj_matrix * (f1 + f2.T)
        unnormalized_attentions = (F.sigmoid(attn_logits) - 0.5).to_sparse()
        attn_probs = torch.sparse.softmax(unnormalized_attentions / self.temp, dim=1)
        attn_probs = attn_probs.to_dense()
        node_feats = torch.matmul(attn_probs, node_feats)



        return node_feats

class GATMultiHead(nn.Module):

    def __init__(self,
                 c_in,  # dimensionality of input features
                 c_out,  # dimensionality of output features
                 n_heads=1,  # number of attention heads
                 concat_heads=True,  # concatenate attention heads or not
                 ):

        super().__init__()

        self.n_heads = n_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % n_heads == 0, "The number of output features should be divisible by the number of heads."
            c_out = c_out // n_heads

        self.block = nn.ModuleList()
        for i_block in range(self.n_heads):
            self.block.append(GATSingleHead(c_in=c_in, c_out=c_out))

    def forward(self,
                node_feats,  # input node features
                adj_matrix,  # adjacency matrix including self-connections
                ):

        res = []
        for i_block in range(self.n_heads):
            res.append(self.block[i_block](node_feats, adj_matrix))

        if self.concat_heads:
            node_feats = torch.cat(res, dim=1)
        else:
            node_feats = torch.mean(torch.stack(res, dim=0), dim=0)

        return node_feats

class DeconvNet(nn.Module):

    def __init__(self,
                 hidden_dims,  # dimensionality of hidden layers
                 n_celltypes,  # number of cell types
                 n_heads,  # number of attention heads
                 rate1,
                 rate2,
                 ):
        super().__init__()

        # define layers
        # encoder layers
        self.encoder_layer1 = GATMultiHead(hidden_dims[0], hidden_dims[1], n_heads=n_heads, concat_heads=True)
        self.encoder_layer2 = DenseLayer(hidden_dims[1], hidden_dims[2])

        # decoder layers
        self.decoder_layer1 = GATMultiHead(hidden_dims[2], hidden_dims[1], n_heads=n_heads, concat_heads=True)
        self.decoder_layer2 = DenseLayer(hidden_dims[1], hidden_dims[0])

        # deconvolution layers
        self.deconv_beta_layer = DenseLayer(hidden_dims[2], n_celltypes, zero_init=False)

        self.rate1 = rate1
        self.rate2 = rate2

        # self.p = nn.Parameter(torch.Tensor(1, 1).zero_())

        self.p = nn.Parameter(torch.Tensor(1, 1))
        nn.init.constant_(self.p, 0.01)

    def forward(self,
                adj_matrix,  # adjacency matrix including self-connections
                node_feats,  # input node features
                count_matrix,  # gene expression counts
                library_size,  # library size (based on Y)
                basis,  # basis matrix
                ):
        # encoder
        Z = self.encoder(adj_matrix, node_feats)

        # deconvolutioner
        beta = self.deconvolutioner(Z)

        # decoder
        node_feats_recon = self.decoder(adj_matrix, Z)

        # reconstruction loss of node features
        self.features_loss = torch.mean(torch.sqrt(torch.sum(torch.pow(node_feats - node_feats_recon, 2), axis=1)))

        # deconvolution loss
        log_lam = torch.log(torch.matmul(beta, basis) + 1e-6)
        lam = torch.exp(log_lam)

        self.decon_loss_1 = - torch.mean(torch.sum(count_matrix *
                                                   (torch.log(self.p + 1e-6) + torch.log(library_size + 1e-6) + log_lam) - self.p * library_size * lam,
                                                   axis=1))


        loss = self.decon_loss_1

        return loss

    def evaluate(self, adj_matrix, node_feats):
        # encoder
        Z = self.encoder(adj_matrix, node_feats)

        # deconvolutioner
        beta = self.deconvolutioner(Z)

        return Z, beta, self.p

    def encoder(self, adj_matrix, node_feats):
        H = node_feats
        H = F.elu(self.encoder_layer1(H, adj_matrix))
        Z = self.encoder_layer2(H)
        return Z

    def decoder(self, adj_matrix, Z):
        H = F.elu(self.decoder_layer1(Z, adj_matrix))
        X_recon = self.decoder_layer2(H)
        return X_recon

    def deconvolutioner(self, Z):
        beta = self.deconv_beta_layer(F.elu(Z))
        beta = F.softmax(beta, dim=1)
        H = F.elu(Z)
        return beta

