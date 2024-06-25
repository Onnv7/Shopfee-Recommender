from .recommend_model import GNNModel
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import dgl
from dgl.dataloading import MultiLayerNeighborSampler, as_edge_prediction_sampler
from torch.multiprocessing import Pool, Process, set_start_method
import pickle

class GNNUtils :
    embeddings = None
    model = None
    graph = None
    # nodeLoad_test = None
    user_id_map = None
    product_id_map = None
    def __init__(self):
        
        out_dim = 32 
        cuda = torch.cuda.is_available()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        GNNUtils.model, GNNUtils.graph, GNNUtils.user_id_map, GNNUtils.product_id_map = self.load()    
        
        
        GNNUtils.embeddings = self.calculate_and_save_full_graph_embeddings()
        # with open('./config/embeddings.pkl', 'rb') as f:
        #     GNNUtils.embeddings = pickle.load(f)
        
        print("inited", GNNUtils.model)
        print("inited graph", GNNUtils.graph)
        pass
    
    @staticmethod
    def load(): 
        out_dim = 64 
        hidden_dim = 256
        prediction = 'cos' 
        aggregator_hetero = 'mean' 
        dropout = 0.3 
        embedding_layer = True

        validation_graph, _ = dgl.load_graphs('./config/graph.dgl')
        validation_graph = validation_graph[0]

        dim_dict = {'user': validation_graph.nodes['user'].data['features'].shape[1],
                    'product': validation_graph.nodes['product'].data['features'].shape[1],
                    'out':  out_dim,
                    'hidden':hidden_dim}

        trained_model = GNNModel(validation_graph, 3, dim_dict, dropout, prediction, aggregator_hetero, embedding_layer)

        # Load trọng số đã lưu vào mô hình khởi tạo
        trained_model.load_state_dict(torch.load('./config/model.pth', map_location=torch.device('cpu')))

        # Chuyển mô hình sang thiết bị cần thiết (nếu cần)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trained_model.to(device)
        
        user_id_map = pd.read_csv('./config/user_id_table.csv')
        product_id_map = pd.read_csv('./config/product_id_table.csv')
        return trained_model, validation_graph, user_id_map, product_id_map
    def update_edge_rating(self, src_id, dst_id, new_rating, etype):
        # Find edge id between user_id and product_id
        try:
            eid = GNNUtils.graph.edge_ids(torch.tensor([src_id]), torch.tensor([dst_id]), etype=etype)
            if eid is not None:
            # Update edge rating
                GNNUtils.graph.edata['rating'][eid] = new_rating
                return True
            else:
                return False
        except Exception as e:
            print(e)
            return False
    

    def add_new_edge(self, user_id, product_id, rating):
        
        user_id_embedded = self.getMappedUserId(user_id)
        product_id_embedded = self.getMappedProductId(product_id)
        edge_feature = torch.tensor([rating])
        if self.update_edge_rating(user_id_embedded, product_id_embedded, edge_feature, 'rating') and self.update_edge_rating(product_id_embedded, user_id_embedded, edge_feature, 'bought-by'):
            pass
        else:
            GNNUtils.graph.add_edges(user_id_embedded, product_id_embedded,  {'rating': edge_feature}, 'rating')
            GNNUtils.graph.add_edges(product_id_embedded, user_id_embedded,  {'rating': edge_feature}, 'bought-by')
        
        GNNUtils.embeddings = self.calculate_and_save_full_graph_embeddings()
        GNNUtils.user_id_map.to_csv('./config/user_id_table.csv')
        dgl.save_graphs('./config/graph.dgl', GNNUtils.graph)
        return True

    def recommend(self, user_id, k):
        user_emb = GNNUtils.embeddings['user'][user_id]
        
        user_emb_rpt = user_emb.repeat(GNNUtils.graph.num_nodes('product'), 1)
        
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        ratings = cos(user_emb_rpt, GNNUtils.embeddings['product'])
        
        ratings_formatted = ratings.cpu().detach().numpy().reshape(GNNUtils.graph.num_nodes('product'),)
        order = np.argsort(-ratings_formatted)
        
        rec = order[:k]  # top k recommendations
        return rec
    
    def recommend_for_user(self, user_id, k):
        mapped_user_id = self.getMappedUserId(user_id)
            
        recommended_products = self.recommend(mapped_user_id, k)
        result = self.getRealProductIds(recommended_products)
        print(user_id, result, mapped_user_id, recommended_products, sep=' - ')
        return result
    
    def add_new_user_feat(self, user_id, gender):
        if user_id in GNNUtils.user_id_map['UserID'].values:
            return False
        
        max_user_id = GNNUtils.user_id_map['user_new_id'].max() if not GNNUtils.user_id_map.empty else 0
        GNNUtils.user_id_map.loc[len(GNNUtils.user_id_map)] = [user_id, max_user_id + 1]
        
        if gender == 'FEMALE':
            user_feature = torch.tensor([1, 0])
        elif gender == 'MALE':
            user_feature = torch.tensor([0, 1])
        else:
            user_feature = torch.tensor([0, 0])
        
        temple = GNNUtils.graph.ndata['features']['user'].clone()
        GNNUtils.graph = dgl.add_nodes(GNNUtils.graph, 1, ntype='user')
        GNNUtils.graph.nodes['user'].data['features'] = torch.cat((temple, user_feature.unsqueeze(0)), dim=0)
        
        GNNUtils.embeddings = self.calculate_and_save_full_graph_embeddings()
        
        GNNUtils.user_id_map.to_csv('./config/user_id_table.csv')
        dgl.save_graphs('./config/graph.dgl', GNNUtils.graph)
        
        return True
    
    
    def calculate_and_save_full_graph_embeddings(self):
        # Ensure the graph is on the correct device
        cuda = torch.cuda.is_available()
        device = torch.device('cuda') if cuda else torch.device('cpu')
        GNNUtils.graph = GNNUtils.graph.to(device)
        
        h = GNNUtils.graph.ndata['features']
            
        if True:
            # Assuming 'user' and 'product' are keys in input_features
            h['user'] = GNNUtils.model.user_embed(h['user'].to(device))
            h['product'] = GNNUtils.model.item_embed(h['product'].to(device))
        
        # Calculate representations
        with torch.no_grad():  
            for i in range(len(GNNUtils.model.layers)):
                h = GNNUtils.model.layers[i](GNNUtils.graph, h)
        with open('./config/embeddings.pkl', 'wb') as f:
            pickle.dump(h, f)
        return h
    def getRealUserId(self, mappedId):
        result = GNNUtils.user_id_map.loc[GNNUtils.user_id_map['user_new_id'] == mappedId, 'UserID']
        
        if not result.empty:
            return result.iloc[0]
        else:
            return None 

    def getMappedUserId(self, realId):
        result = GNNUtils.user_id_map.loc[GNNUtils.user_id_map['UserID'] == realId, 'user_new_id']
        
        if not result.empty:
            return result.iloc[0]
        else:
            return None 
        
    def getRealProductId(self, mappedId):
        result = GNNUtils.product_id_map.loc[GNNUtils.product_id_map['product_new_id'] == mappedId, 'ItemID']
        
        if not result.empty:
            return result.iloc[0]
        else:
            return None 

    def getMappedProductId(self, realId):
        result = GNNUtils.product_id_map.loc[GNNUtils.product_id_map['ItemID'] == realId, 'product_new_id']
        
        if not result.empty:
            return result.iloc[0]
        else:
            return None 
        
    def getRealProductIds(self, recommended_products):
        mapped_ids = []
        for id in recommended_products:
            mapped_id = self.getRealProductId(id)
            if mapped_id is not None:
                mapped_ids.append(mapped_id)
        return mapped_ids
    
    
    