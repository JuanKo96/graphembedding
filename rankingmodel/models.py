import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

class SequentialEmbedding(nn.Module):
    '''Nothing much than a placeholder class for now.
    '''
    def __init__(self, input_size, output_size, device):
        super(SequentialEmbedding, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.device = device

        # mock model architecture:
        self.linear = torch.nn.Linear(input_size, output_size).to(device)
        self.activation_function = nn.LeakyReLU().to(device)

    def forward(self, input_data):
        '''
        Replace this stupid linear layer with LSTMs
        '''
        out = self.linear(input_data)
        out = self.activation_function(out)
        return out
    
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

        N = seq_embed.size(0)           # number of stocks
        U = seq_embed.size(1)           # sequential embedding size
        K = self.rel_encoding.size(-1)  # multi-hot binary graph encoding size.
        for k in range(1,self.k_hops+1):            
            # [N x U] => [(N x N) x U] => [N x N x U] => [N x N x (U + U)]
            seq_repeated = seq_embed.repeat_interleave(repeats=N, dim=0)
            seq_repeated = seq_repeated.reshape((N, N, U)) 
            seq_combined = torch.cat((
                seq_repeated,                   
                seq_repeated.transpose(0,1),
            ), dim=-1).to(self.device)

            # combined[i][j] = [e_i, e_j, a_ij]
            combined = torch.cat((
                seq_combined,
                self.rel_encoding
            ), dim=-1).to(self.device)
            assert combined.size() == (N, N, U+U+K)

            # set unconnected nodes to zero.
            mask_dim_expand = self.rel_mask.unsqueeze(dim=-1).to(self.device)
            combined = mask_dim_expand.mul(combined)
            
            # weights[i][j] = g(e_i, e_j, a_ij) values from Temporal paper.
            weights = combined
            for one_layer_k in range(1, self.hop_layers+1):
                linear_layer = getattr(self, f"linear_{k}_hop_layer_{one_layer_k}")
                weights = linear_layer(weights)
                weights = self.activation_function(weights)
            assert weights.size() == (N, N, 1)

            # mask out disconnected nodes again
            weights = mask_dim_expand.mul(weights)
            weights = weights.squeeze()
            assert weights.size() == (N, N)
            
            # refer to Temporal paper page 9. d_j = number of nodes satisfying sum(a_ij) > 0
            D = self.rel_mask.sum(dim=-1)
            scaled_weights = weights / D.unsqueeze(dim=-1)
            scaled_weights = scaled_weights.unsqueeze(dim=-1)
            assert scaled_weights.size() == (N, N, 1)
            
            # weighted_embeds[i][j] = weight_j * [e_i, e_j]
            weighted_embeds = scaled_weights.mul(seq_combined)
            assert weighted_embeds.size() == (N, N, U+U)

            # weighted_neigh_embeds[i][j] = weight_j * e_j
            weighted_neigh_embeds = weighted_embeds.split(U, dim=-1)[1]
            assert weighted_neigh_embeds.size() == (N, N, U)

            # relational_embed[i] = weight_x * e_x + weight_y* e_y 
            # where weight_x + weight_y = 1
            seq_embed = torch.sum(weighted_neigh_embeds, dim=0)
            assert seq_embed.size() == (N, U)
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
    def __init__(self, input_size, seq_embed_size, relational_encoding, k_hops, hop_layers, device):
        super(TemporalSAGE, self).__init__()
        self.input_size = input_size
        self.seq_embed_size = seq_embed_size
        self.relational_encoding = relational_encoding.to(device)
        self.k_hops = k_hops
        self.hop_layers = hop_layers
        self.device = device

        relational_mask = relational_encoding.sum(dim=-1).to(device)
        relational_mask = torch.where(relational_mask > 0, 1, 0)
        
        self.sequential_embedding_model = SequentialEmbedding(
            input_size=input_size, 
            output_size=seq_embed_size,
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
            layers=3,
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
        return predictions

# def __init__(self, seq_embeds, device, rel_encoding, rel_mask, k_hops, implicit_modeling=True):

def test_sage():
    N = 200 # number of stocks
    K = 2 # number of possible relations in graph
    seq_embed_size = 64
    input_size = 100 # completely arbitrary 
    device = "cuda:0"

    input_data = torch.rand(N, input_size).to(device)

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
        device=device
    )

    predictions = model(input_data)
    assert predictions.size() == (N, 1)
    print(model)

if __name__ == "__main__":
    test_sage()


        

        
                

