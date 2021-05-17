import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

class RelationalEmbedding(nn.Module):
    def __init__(self, seq_embed_size, device, rel_encoding, rel_mask, k_hops, layers_k, implicit_modeling=True):
        super(RelationalEmbedding, self).__init__()

        self.seq_embed_size = seq_embed_size
        self.device = device
        self.rel_encoding = rel_encoding
        self.rel_mask = rel_mask
        self.k_hops = k_hops
        self.layers_k = layers_k
        self.implicit_modeling = True
        self.softmax_dim_1 = torch.nn.Softmax(dim=1)

        # Dummy checks
        if self.layers_k < 1:
            loger.error("layers_k must be >= 1.")
            raise Exception

        if self.implicit_modeling:
            input_size = 2*seq_embed_size + rel_encoding.size(-1) # U + U + K
            for k in range(1,k_hops+1):
                for one_layer_i in range(1, layers_k+1):
                    if one_layer_i != layers_k:
                        one_layer = nn.Linear(input_size, input_size)
                    else:
                        one_layer = nn.Linear(input_size, 1)
                    setattr(self, f"linear_{k}_hop_layer_{one_layer_i}", one_layer)
    
    def forward(self, seq_embed):
        """
        Parameters:
            seq_embed (Tensor size=(N, U)): sequential embedding matrix. N = len(stocks), U = seq embedding size. 
        Returns:
            relational_embeddings (Tensor size=(N, U))

        Felt reluctant about single-letter variable names but code became much more readable like this,
        especially when these are widely-agreed upon names.
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
            ), dim=-1)

            # combined[i][j] = [e_i, e_j, a_ij]
            combined = torch.cat((
                seq_combined,
                self.rel_encoding               
            ), dim=-1)
            assert combined.size() == (N, N, U+U+K)

            # set unconnected nodes to zero.
            mask_dim_expand = self.rel_mask.unsqueeze(dim=-1)
            combined = mask_dim_expand.mul(combined)
            
            # weights[i][j] = g(e_i, e_j, a_ij) values from Temporal paper.
            weights = combined
            for one_layer_k in range(1, self.layers_k+1):
                linear_layer = getattr(self, f"linear_{k}_hop_layer_{one_layer_k}")
                weights = linear_layer(weights)
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



# def __init__(self, seq_embeds, device, rel_encoding, rel_mask, k_hops, implicit_modeling=True):

def test_forward():
    K = 2
    seq_embeds = torch.Tensor([
        [1,1,1,1,1],
        [2,2,2,2,2],
        [3,3,3,3,3]
    ])*0.1
    # this creates a logically flawed rel_encoding and is source of unnecessary error in certain cases. edit this tomorrow
    rel_encoding = torch.randint(0,2, (seq_embeds.size(0), seq_embeds.size(0), K))
    int_encoding = rel_encoding.sum(dim=-1)
    rel_mask = torch.where(int_encoding > 0, 1, 0)

    model = RelationalEmbedding(
        seq_embed_size=seq_embeds.size(-1),
        device="cuda:0",
        rel_encoding=rel_encoding,
        rel_mask=rel_mask,
        k_hops=3,
        layers_k=2,
        implicit_modeling=True
    )
    relational_embedding = model.forward(seq_embeds)
    

if __name__ == "__main__":
    test_forward()


        

        
                

