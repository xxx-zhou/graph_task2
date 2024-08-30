
from torch.nn import Sequential as Seq, Linear, ReLU
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import train_test_split_edges

# 定义一个通用的训练函数
def train_model(model, data, optimizer, device, use_sampler=False):
    model.train()
    total_loss = 0
    for epoch in range(1, 201):
        model.zero_grad()
        if use_sampler:
            # 使用NeighborSampler进行训练
            sampler = NeighborSampler(data.train_pos_edge_index, sizes=[15, 10], batch_size=128, shuffle=True)
            for batch_size, n_id, adjs in sampler:
                adjs = [adj.to(device) for adj in adjs]
                # 将adjs转换为edge_index格式
                edge_index = adjs[0].edge_index  # 获取第一个邻接矩阵的边索引
                out = model(data.x[n_id].to(device), edge_index)
                loss = F.nll_loss(out[batch_size:], data.y[n_id][batch_size:])
                loss.backward()
                total_loss += loss.item() * out.size(0)
        else:
            # 正常训练
            out = model(data.x.to(device), data.train_pos_edge_index.to(device))
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            total_loss += loss.item() * out.size(0)
        optimizer.step()
    return total_loss / data.y.size(0)



# 定义不同的图神经网络模型
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return self.conv2(x, edge_index)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def forward(self, data):
        z = self.encode(data.x, data.train_pos_edge_index)
        return self.decode(z, data.test_pos_edge_index, data.test_neg_edge_index)


class GAT(torch.nn.Module): # Removed duplicate class definition
    def __init__(self, num_features, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, num_classes, heads=1, concat=True, dropout=0.6)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def forward(self, data):
        z = self.encode(data.x, data.train_pos_edge_index)
        return self.decode(z, data.test_pos_edge_index, data.test_neg_edge_index)


# ... GAT类的实现与前面相同 ...

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, 128)
        self.conv2 = SAGEConv(128, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


# ... GraphSAGE类的实现 ...

class GIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GIN, self).__init__()
        nn = Seq(Linear(in_channels, 64), ReLU(), Linear(64, 64))
        self.conv1 = GINConv(nn)
        self.conv2 = GINConv(nn)
        self.fc = Linear(64, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


# ... GIN类的实现 ...

# 数据集名称列表
datasets = ['Cora', 'Citeseer']

# 测试不同的模型
model_types = {
    'GCN': GCN,
    'GAT': GAT,
    'GraphSAGE': GraphSAGE,
    'GIN': GIN
}


if torch.cuda.is_available():
    print("CUDA is available!")
    device = torch.device('cuda')
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device('cpu')

for dataset_name in datasets:
    print(f"\nTesting on {dataset_name} dataset")
    dataset = Planetoid(root='/tmp', name=dataset_name)
    data = dataset[0]
    data = train_test_split_edges(data, test_ratio=0.05)

    for model_name, model_cls in model_types.items():
        print(f"\nUsing {model_name} model")
        model = model_cls(dataset.num_features, dataset.num_classes)
        model = model.to(torch.device('cuda'))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # 训练模型
        train_model(model, data, optimizer, torch.device('cuda'), use_sampler=True)