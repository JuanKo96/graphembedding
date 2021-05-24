import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

class SequentialEmbedding(nn.Module):
    def __init__(self, input_shape, seq_embed_size, lstm_layers, device="cpu"):
        super(SequentialEmbedding, self).__init__()
        
        batch_size, window_size, N, n_features = input_shape

        self.batch_size = batch_size
        self.window_size = window_size
        self.N = N
        self.n_features = n_features
        self.seq_embed_size = seq_embed_size
        self.device = device
        self.lstm_layers = lstm_layers

        self.lstm = torch.nn.LSTM(n_features, seq_embed_size, num_layers=lstm_layers, batch_first=True).to(device)
        print(self.lstm)
    def forward(self, input_data):
        '''
        Here was a source of a huge logic bug confusion.
        Say N = 736, batch_size = 16, window_size = 30.

        Then input_data.size() == (16, 30, 736) == (batch_size, seq, features???????)
        and out.size() became == (16, 64), suggesting:
            out[0] == "seq embedding" for (30, 736).

        which isn't what we want! if you think about it, its more like:
            we have 16 rows in a batch and 736 stocks in each row, and a single-valued feature
            for each stock.

        so we actually need to do
        (16, 30, 736) => (16 * 736, 30) => LSTM => (16 * 736, 64) => (16, 736, 64)

        and we actually should work with (16, 30, 736, 1) instead of (16, 30, 736) so that:
            1. we avoid future confusion by dividing dimensions with function:
                - being able to say: [batch_size, window_size, N, n_features]
            2. are we really only going to use one feature per stock???
        '''
        input_data = input_data.permute(0, 2, 1, 3) # (16, 30, 736, 1) => (16, 736, 30, 1)
        input_data = input_data.reshape([self.batch_size * self.N, self.window_size, self.n_features]) # => (16*736, 30, 1)
        
        out, _ = self.lstm(input_data) # (16*736, 30, 1) => (16 * 736, 64)
        out = out.reshape(self.batch_size, self.N, self.window_size, self.seq_embed_size) # => (16, 736, h_states=30, 64)
        
        return out[:, :,-1,:] # (batch_size, N, U)
    
class RelationalEmbedding(nn.Module):
    def __init__(self, seq_embed_size, rel_encoding, rel_mask, k_hops, hop_layers, device):
        super(RelationalEmbedding, self).__init__()

        self.seq_embed_size = seq_embed_size
        self.rel_encoding = rel_encoding.to(device)
        self.rel_mask = rel_mask.to(device)
        self.k_hops = k_hops
        self.hop_layers = hop_layers
        self.device = device

        self.softmax_dim_1 = torch.nn.Softmax(dim=1).to(device)
        self.activation_function = nn.LeakyReLU().to(device)

        # Dummy checks
        if self.hop_layers <= 0:
            loger.error("hop_layers must be >= 1.")
            raise Exception

        # Making FC layers: linear_k_hop_layer_i
        input_size = 2*seq_embed_size + rel_encoding.size(-1) # U + U + K
        for k in range(1,k_hops+1):
            for one_layer_i in range(1, hop_layers+1):
                if one_layer_i != hop_layers:
                    one_layer = nn.Linear(input_size, input_size).to(device)
                else:
                    one_layer = nn.Linear(input_size, 1).to(device)
                setattr(self, f"linear_{k}_hop_layer_{one_layer_i}", one_layer)

    def forward(self, seq_embed):
        """
        Parameters:
            seq_embed (Tensor size=(N, U)): sequential embedding matrix. N = len(stocks), U = seq embedding size. 
        Returns:
            relational_embeddings (Tensor size=(N, U))

        Felt reluctant about single-letter variable names but code became much more readable like this,
        especially when these are widely-agreed upon names.
        N = # of stocks
        U = sequential embedding size
        K = # of relation that pair a given two nodes

        The following is a walkthrough of one of the more confusing part of the code:

        Suppose seq_embed = [1,2,3] 

        seq_repeated = 
        | 1 1 1 |
        | 2 2 2 |
        | 3 3 3 |

        seq_repeated.transpose(0,1)=
        | 1 2 3 |
        | 1 2 3 |
        | 1 2 3 |

        seq_combined=
        | [1,1] [1,2] [1,3] |
        | [2,1] [2,2] [2,3] |
        | [3,1] [3,2] [3,3] |

        hence, seq_combined[i][j] = [e_i, e_j]
        where e_k = sequential embedding of kth stock.
        """
        batch_size, N, U = seq_embed.size()
        K = self.rel_encoding.size(-1)  # multi-hot binary graph encoding size.
        for k in range(1,self.k_hops+1):            
            # [N x U] => [(N x N) x U] => [N x N x U] => [N x N x (U + U)]
            encoding_repeated = self.rel_encoding.unsqueeze(dim=0).expand(batch_size, -1, -1, -1) # make it into a batch of size 1
            seq_repeated = seq_embed.repeat_interleave(repeats=N, dim=1)
            seq_repeated = seq_repeated.reshape((batch_size, N, N, U)) 
            seq_combined = torch.cat((
                seq_repeated,                   
                seq_repeated.transpose(1,2),
            ), dim=-1).to(self.device)
            # combined[i][j] = [e_i, e_j, a_ij]
            combined = torch.cat((
                seq_combined,
                encoding_repeated
            ), dim=-1).to(self.device)
            assert combined.size() == (batch_size, N, N, U+U+K)

            # set unconnected nodes to zero.
            mask_dim_expand = self.rel_mask.unsqueeze(dim=-1).to(self.device)
            combined = mask_dim_expand.mul(combined)
            
            # weights[i][j] = g(e_i, e_j, a_ij) values from Temporal paper.
            weights = combined
            for one_layer_k in range(1, self.hop_layers+1):
                linear_layer = getattr(self, f"linear_{k}_hop_layer_{one_layer_k}")
                weights = linear_layer(weights)
                weights = self.activation_function(weights)
            assert weights.size() == (batch_size, N, N, 1)

            # mask out disconnected nodes again
            weights = mask_dim_expand.mul(weights)
            weights = weights.squeeze()
            assert weights.size() == (batch_size, N, N)
            
            # refer to Temporal paper page 9. d_j = number of nodes satisfying sum(a_ij) > 0
            D = self.rel_mask.sum(dim=-1)
            scaled_weights = weights / D.unsqueeze(dim=-1)
            scaled_weights = scaled_weights.unsqueeze(dim=-1)
            assert scaled_weights.size() == (batch_size, N, N, 1)
            
            # weighted_embeds[i][j] = weight_j * [e_i, e_j]
            weighted_embeds = scaled_weights.mul(seq_combined)
            assert weighted_embeds.size() == (batch_size, N, N, U+U)

            # weighted_neigh_embeds[i][j] = weight_j * e_j
            weighted_neigh_embeds = weighted_embeds.split(U, dim=-1)[1]
            assert weighted_neigh_embeds.size() == (batch_size, N, N, U)

            # relational_embed[i] = weight_x * e_x + weight_y* e_y 
            # where weight_x + weight_y = 1
            seq_embed = torch.sum(weighted_neigh_embeds, dim=1)
            assert seq_embed.size() == (batch_size, N, U)
        return seq_embed

class FullyConnected(nn.Module):
    def __init__(self, input_size, output_size, layers, device):
        super(FullyConnected, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
        self.device = device
        self.activation_function = nn.LeakyReLU().to(device)
        for one_layer in range(1, layers+1):
            if one_layer != layers:
                setattr(self, f"linear_layer_{one_layer}", nn.Linear(input_size, input_size).to(device))
            else:
                setattr(self, f"linear_layer_{layers}", nn.Linear(input_size, output_size).to(device))

    def forward(self, combined_embeddings):
        for one_layer in range(1, self.layers+1):
            linear_layer = getattr(self, f"linear_layer_{one_layer}")
            combined_embeddings = linear_layer(combined_embeddings)
            combined_embeddings = self.activation_function(combined_embeddings)
        return combined_embeddings


class TemporalSAGE(nn.Module):
    def __init__(self, input_shape, seq_embed_size, relational_encoding, k_hops, hop_layers, fn_layers, lstm_layers, device):
        super(TemporalSAGE, self).__init__()
        self.input_shape = input_shape
        self.seq_embed_size = seq_embed_size
        self.relational_encoding = relational_encoding.to(device)
        self.k_hops = k_hops
        self.hop_layers = hop_layers
        self.lstm_layers = lstm_layers
        self.fn_layers = fn_layers
        self.device = device

        relational_mask = relational_encoding.sum(dim=-1).to(device)
        relational_mask = torch.where(relational_mask > 0, 1, 0)
        
        self.sequential_embedding_model = SequentialEmbedding(
            input_shape=input_shape, 
            seq_embed_size=seq_embed_size,
            lstm_layers=lstm_layers,
            device=device
        ).to(device)
        self.relational_embedding_model = RelationalEmbedding(
            seq_embed_size=seq_embed_size, 
            rel_encoding=relational_encoding, 
            rel_mask=relational_mask, 
            k_hops=k_hops, 
            hop_layers=hop_layers,
            device=device
        ).to(device)
        self.combined_prediction_model = FullyConnected(
            input_size=2*seq_embed_size, 
            output_size=1, 
            layers=self.fn_layers,
            device=device
        ).to(device)


    def forward(self, input_data):
        seq_embeddings = self.sequential_embedding_model(input_data)
        relational_embeddings = self.relational_embedding_model(seq_embeddings)
        combined_embeddings = torch.cat((
            seq_embeddings,
            relational_embeddings
        ), dim=-1)
        predictions = self.combined_prediction_model(combined_embeddings)
        predictions = predictions.squeeze(dim=-1)

        return predictions

# def __init__(self, seq_embeds, device, rel_encoding, rel_mask, k_hops, implicit_modeling=True):

def test_sage():
    N = 200 # number of stocks
    K = 2 # number of possible relations in graph
    seq_embed_size = 64
    input_size = 100 # completely arbitrary 
    window_size = 30 # random
    device = "cuda:0"

    input_data = torch.rand(N, window_size,input_size).to(device)

    rel_encoding = torch.randint(0,2, (N, N, K))
    for i in range(rel_encoding.size(0)):
        for j in range(rel_encoding.size(1)):
            rel_encoding[i][j] = rel_encoding[j][i]

    int_encoding = rel_encoding.sum(dim=-1)
    rel_mask = torch.where(int_encoding > 0, 1, 0)

    model = TemporalSAGE(
        input_size=input_size, 
        seq_embed_size=seq_embed_size, 
        relational_encoding=rel_encoding, 
        k_hops=3, 
        hop_layers=2, 
        fn_layers=fn_layers,
        device=device
    )

    predictions = model(input_data)
    assert predictions.size() == (N, 1)
    print(model)

if __name__ == "__main__":
    test_sage()
