import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import dgl.function as dgl_func
import torch.nn.functional as torch_nn_func
import dgl.function as fn

class CosinePrediction(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, graph, h):
        with graph.local_scope():
            for edge_type in graph.canonical_etypes:
                try:
                    graph.nodes[edge_type[0]].data['norm_h'] = torch_nn_func.normalize(h[edge_type[0]], p=2, dim=-1)
                    graph.nodes[edge_type[2]].data['norm_h'] = torch_nn_func.normalize(h[edge_type[2]], p=2, dim=-1)
                    graph.apply_edges(fn.u_dot_v('norm_h', 'norm_h', 'cos'), etype=edge_type)
                except ValueError:
                   print("Cosine similarity fucntion is not correct!")
            ratings = graph.edata['cos']
        return ratings
    
class MessagePassing(nn.Module):    
      def __init__(self, input_features, output_features, dropout,):
        super().__init__()
        self._in_neigh_feats, self._in_self_feats = input_features
        self._output_features = output_features 
        self.dropout = nn.Dropout(dropout)
        self.fc_self = nn.Linear(self._in_self_feats, output_features, bias=False)
        self.fc_neighbour = nn.Linear(self._in_neigh_feats, output_features, bias=False)
        self.fc_pre_agg = nn.Linear(self._in_neigh_feats, self._in_neigh_feats, bias=False)
      
      def forward(self, graph, x):
        
        h_neighbour, h_self = x
        h_self = self.dropout(h_self)
        h_neighbour = self.dropout(h_neighbour)
        

        graph.srcdata['h'] = torch_nn_func.relu(self.fc_pre_agg(h_neighbour))
        graph.update_all(dgl_func.copy_u('h', 'm'), dgl_func.mean('m', 'neigh'))
        h_neighbour = graph.dstdata['neigh']

        #message passing
        z = self.fc_self(h_self) + self.fc_neighbour(h_neighbour)
        z = torch_nn_func.relu(z)

        z_normalization = z.norm(2, 1, keepdim=True)
        z_normalization = torch.where(z_normalization == 0, torch.tensor(1.).to(z_normalization), z_normalization)
        z = z / z_normalization

        return z
    
class NodeEmbedding(nn.Module):
    def __init__(self, input_features, output_features,):
        super().__init__()
        self.project_features = nn.Linear(input_features, output_features)

    def forward(self, node_features):
        return self.project_features(node_features)
    
class GNNModel(nn.Module):

    def __init__(self, g, n_layers: int, dim_dict, dropout, pred, aggregator_hetero, embedding_layer,):

        super().__init__()
        self.embedding_layer = embedding_layer

        if embedding_layer:
            self.user_embed = NodeEmbedding(dim_dict['user'], dim_dict['hidden'])
            self.item_embed = NodeEmbedding(dim_dict['product'], dim_dict['hidden'])

        self.layers = nn.ModuleList()

        # input layer
        if not embedding_layer:
            self.layers.append(
                dglnn.HeteroGraphConv(
                    {etype[1]: MessagePassing((dim_dict[etype[0]], dim_dict[etype[2]]), dim_dict['hidden'], dropout) for etype in g.canonical_etypes}, 
                    aggregate=aggregator_hetero)
                    )

        # hidden layers
        for i in range(n_layers - 2):
            self.layers.append(
                dglnn.HeteroGraphConv(
                    {etype[1]: MessagePassing((dim_dict['hidden'], dim_dict['hidden']), dim_dict['hidden'], dropout) for etype in g.canonical_etypes},
                    aggregate=aggregator_hetero)
                    )

        # output layer
        self.layers.append(
            dglnn.HeteroGraphConv(
                {etype[1]: MessagePassing((dim_dict['hidden'], dim_dict['hidden']), dim_dict['out'], dropout) for etype in g.canonical_etypes}, 
                aggregate=aggregator_hetero)
                )
        self.pred_fn = CosinePrediction()
        

    def get_repr(self, blocks, h):

        for i in range(len(blocks)):         
            layer = self.layers[i]
            h = layer(blocks[i], h)
          
        return h

    def forward(self, blocks, h, pos_g, neg_g, embedding_layer: bool=True, ):
        if embedding_layer:
            h['user'] = self.user_embed(h['user'])
            h['product'] = self.item_embed(h['product'])

        h = self.get_repr(blocks, h)
        pos_score = self.pred_fn(pos_g, h)
        neg_score = self.pred_fn(neg_g, h)

        return h, pos_score, neg_score