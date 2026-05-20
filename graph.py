import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


class GraphConvolutionLayer(nn.Module): # GraphConvolutionLayer 类实现了一个基本的图卷积操作，它考虑了邻接节点的信息，并通过残差连接和激活函数来更新节点的特征表示。
    def __init__(self, input_size, hidden_size, graph_drop):
        super(GraphConvolutionLayer, self).__init__()
        self.W = nn.Parameter(torch.Tensor(size=(input_size, hidden_size))) # 创建一个可训练的权重参数 W。
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu')) # 使用 Xavier 初始化方法初始化 W，这种方法通常用于保持输入和输出的方差一致，适用于激活函数为ReLU的场景。
        self.bias = nn.Parameter(torch.Tensor(hidden_size)) # 创建一个可训练的偏置参数 bias。
        nn.init.zeros_(self.bias) # 将 bias 初始化为零。

        self.drop = torch.nn.Dropout(p=graph_drop, inplace=False)

    def forward(self, input): # 定义了类的 forward 方法，用于处理输入数据。输入参数 input 包含节点嵌入和邻接矩阵。
        nodes_embed, node_adj = input
        h = torch.matmul(nodes_embed, self.W.unsqueeze(0)) # 对节点嵌入和权重 W 进行矩阵乘法，以得到变换后的节点特征。
        sum_nei = torch.zeros_like(h) # 初始化一个与 h 形状相同的零张量，用于存储邻居信息的累加。
        sum_nei += torch.matmul(node_adj, h) # 通过邻接矩阵和节点特征的乘积来累加邻居信息。
        degs = torch.sum(node_adj, dim=-1).float().unsqueeze(dim=-1) # 计算每个节点的度（即每个节点的邻居数量）。
        norm = 1.0 / degs # 对度进行归一化处理。
        dst = sum_nei * norm + self.bias # 将归一化后的邻居特征和偏置相加。
        out = self.drop(torch.relu(dst)) # 应用ReLU激活函数，并通过 dropout 层。
        return nodes_embed + out, node_adj # 将变换后的节点特征与原始节点嵌入相加（残差连接），并返回更新后的节点嵌入和未更改的邻接矩阵。


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key): # 计算注意力分数的函数
    N_bt, h, N_nodes, _ = query.shape # N_bt 表示批次大小（batch size）。h 表示头的数量，即多头注意力的头数。N_nodes 表示节点的数量。_ 表示张量的最后一个维度的大小，即特征维度。
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    return scores


class MultiHeadDotProductAttention(nn.Module):
    def __init__(self, edges, in_features: int, out_features: int, n_heads: int, dropout=0.0): # 边的类型 edges、输入特征的大小 in_features、输出特征的大小 out_features、注意力头数 n_heads、丢弃率 dropout 等。
        super().__init__() 

        self.h = n_heads
        self.d_k = out_features // n_heads # d_k 表示每个头的输出特征维度。
        self.edges = edges
        self.linear_layers = nn.ModuleList() 
        # linear_layers 是一个包含多个线性层的模块列表。每个边的注意力计算都会用到两个线性层，一个用于计算查询向量 q，另一个用于计算键向量 k。
        for i in range(len(edges)): # 这行代码的目的是为每种边的不同线性变换创建两个相同的线性层，并将它们添加到 linear_layers 列表中。这样，在计算多头自注意力时，每种边都可以使用不同的线性变换。
            self.linear_layers.append(clones(nn.Linear(in_features, out_features), 2))
        self.dropout = nn.Dropout(p=dropout) # dropout 是一个丢弃层，用于在计算注意力权重时进行随机丢弃。

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor): # h（节点的特征）和adj_mat（邻接矩阵）
        N_bt, N_nodes, _ = h.shape # 这行代码获取输入张量h的形状。N_bt代表批次大小，N_nodes代表节点的数量，而_是一个占位符，用于忽略张量形状的第三个维度。
        adj_mat = adj_mat.unsqueeze(1) # 这行代码在adj_mat张量的第二个维度增加一个额外的维度。这通常用于确保张量的维度对齐，以方便后续操作。
        adj_mat = adj_mat.to(h)
        scores = torch.zeros(N_bt, self.h, N_nodes, N_nodes).to(h) #这里初始化一个全零的张量scores，其形状由N_bt, self.h（可能代表注意力头的数量）, N_nodes, 和N_nodes决定。.cuda()将这个张量移至GPU，以加速计算。
        for edge in range(len(self.edges)):
            q, k = [l(x).view(N_bt, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                    zip(self.linear_layers[edge], (h, h))] # self.linear_layers[edge]可能是一个线性层列表，用于将输入张量h转换为query和key。.view()和.transpose()用于重塑和转置张量，以便符合后续操作的要求。
            scores += attention(q, k).masked_fill(adj_mat != edge + 1, 0) # .masked_fill(adj_mat != edge + 1, 0)用于根据邻接矩阵将不相关的注意力分数置零。 
        scores = scores.masked_fill(scores == 0, -1e9) # 将scores中的零值替换为一个非常小的数（-1e9），这是为了在后续的softmax操作中保持数值稳定性。
        scores = self.dropout(scores)
        attn = F.softmax(scores, dim=-1) # 应用softmax函数来规范化注意力分数，确保每个节点的分数和为1。
        return attn.transpose(0, 1) # 最后，将attn张量的维度进行调整，并将其返回。
class MultiHeadDotProductAttention_2(nn.Module):
    def __init__(self, edges, in_features: int, out_features: int, n_heads: int, dropout=0.0):
        # 调用父类的构造函数
        super().__init__()

        # 保存注意力头的数量
        self.h = n_heads
        # 计算每个头的输出特征维度
        self.d_k = out_features // n_heads
        # 保存边的类型
        self.edges = edges
        # 创建一个模块列表，用于存储线性层
        #self.linear_layers = nn.ModuleList()
        # 为每种边类型创建两个线性层，分别用于计算查询向量 q 和键向量 k
        
        self.linear_layer=clones(nn.Linear(in_features, out_features), 2)
        # 创建一个丢弃层，用于在计算注意力权重时进行随机丢弃
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        # 获取输入张量 h 的形状，N_bt 表示批次大小，N_nodes 表示节点数量
        N_bt, N_nodes, _ = h.shape
        # 在邻接矩阵 adj_mat 的第二个维度增加一个额外的维度，用于维度对齐
        adj_mat = adj_mat.unsqueeze(1)
        # 将邻接矩阵 adj_mat 移动到与输入张量 h 相同的设备上
        adj_mat = adj_mat.to(h)
        # 初始化一个全零的张量 scores，用于存储注意力分数
        scores = torch.zeros(N_bt, self.h, N_nodes, N_nodes).to(h)
        # 遍历每种边类型
        
        q, k = [l(x).view(N_bt, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                zip(self.linear_layer, (h, h))]
            # 计算注意力分数，并根据邻接矩阵将不相关的注意力分数置零
        scores += attention(q, k).masked_fill(adj_mat == 0, 0)  # 修改掩码条件
        # 将 scores 中的零值替换为一个非常小的数（-1e9），以确保在后续的 softmax 操作中数值稳定
        scores = scores.masked_fill(scores == 0, -1e9)
        # 对注意力分数应用丢弃操作
        scores = self.dropout(scores)
        # 应用 softmax 函数对注意力分数进行归一化，确保每个节点的分数和为 1
        attn = F.softmax(scores, dim=-1)
        # 调整注意力矩阵的维度并返回
        return attn.transpose(0, 1)

class AttentionGCNLayer(nn.Module): # AttentionGCNLayer类结合了图卷积网络（GCN）和多头注意力机制，通过多次迭代和注意力加权来提取和处理图结构数据的特征。
    def __init__(self, edges, input_size, nhead=2, graph_drop=0.0, iters=2, attn_drop=0.0): # 边的类型 edges、输入特征的大小 input_size、注意力头数 nhead、图卷积层的迭代次数 iters、图注意力的丢弃率 attn_drop 等。
        super(AttentionGCNLayer, self).__init__() # 这行代码调用超类的构造函数，是在创建子类对象时必须要做的。
        self.nhead = nhead
        self.graph_attention = MultiHeadDotProductAttention(edges, input_size, input_size, self.nhead, attn_drop) # 表示多头自注意力机制。这个实例用于计算节点之间的注意力权重。
        self.gcn_layers = nn.Sequential(
            *[GraphConvolutionLayer(input_size, input_size, graph_drop) for _ in range(iters)]) # 创建一系列图卷积层（GraphConvolutionLayer），数量由iters参数决定，并将它们组合成一个顺序模块（nn.Sequential）。
        self.blocks = nn.ModuleList([self.gcn_layers for _ in range(self.nhead)]) # 创建一个模块列表，每个注意力头都有一个图卷积层的副本。

        self.aggregate_W = nn.Linear(input_size * nhead, input_size) # 定义一个线性层用于聚合来自不同注意力头的输出。

    def forward(self, nodes_embed, node_adj): # 输入参数包括：nodes_embed: 节点的嵌入表示。node_adj: 节点的邻接矩阵。
        output = []
        graph_attention = self.graph_attention(nodes_embed, node_adj) # 调用多头注意力机制实例，计算图的注意力权重。
        for cnt in range(0, self.nhead):
            hi, _ = self.blocks[cnt]((nodes_embed, graph_attention[cnt])) # 对每个注意力头的输出应用图卷积层，并将结果存储在hi中。
            output.append(hi) 
        output = torch.cat(output, dim=-1) # 将所有头的输出沿最后一个维度拼接。
        return self.aggregate_W(output), graph_attention # 通过聚合线性层处理拼接后的输出，并返回最终的输出和计算得到的注意力权重。


class AttentionGCNLayer_2(nn.Module): # AttentionGCNLayer类结合了图卷积网络（GCN）和多头注意力机制，通过多次迭代和注意力加权来提取和处理图结构数据的特征。
    def __init__(self, edges, input_size, nhead=2, graph_drop=0.0, iters=2, attn_drop=0.0): # 边的类型 edges、输入特征的大小 input_size、注意力头数 nhead、图卷积层的迭代次数 iters、图注意力的丢弃率 attn_drop 等。
        super(AttentionGCNLayer_2, self).__init__() # 这行代码调用超类的构造函数，是在创建子类对象时必须要做的。
        self.nhead = nhead
        self.graph_attention = MultiHeadDotProductAttention_2(edges, input_size, input_size, self.nhead, attn_drop) # 表示多头自注意力机制。这个实例用于计算节点之间的注意力权重。
        self.gcn_layers = nn.Sequential(
            *[GraphConvolutionLayer(input_size, input_size, graph_drop) for _ in range(iters)]) # 创建一系列图卷积层（GraphConvolutionLayer），数量由iters参数决定，并将它们组合成一个顺序模块（nn.Sequential）。
        self.blocks = nn.ModuleList([self.gcn_layers for _ in range(self.nhead)]) # 创建一个模块列表，每个注意力头都有一个图卷积层的副本。

        self.aggregate_W = nn.Linear(input_size * nhead, input_size) # 定义一个线性层用于聚合来自不同注意力头的输出。

    def forward(self, nodes_embed, node_adj): # 输入参数包括：nodes_embed: 节点的嵌入表示。node_adj: 节点的邻接矩阵。
        output = []
        graph_attention = self.graph_attention(nodes_embed, node_adj) # 调用多头注意力机制实例，计算图的注意力权重。
        for cnt in range(0, self.nhead):
            hi, _ = self.blocks[cnt]((nodes_embed, graph_attention[cnt])) # 对每个注意力头的输出应用图卷积层，并将结果存储在hi中。
            output.append(hi) 
        output = torch.cat(output, dim=-1) # 将所有头的输出沿最后一个维度拼接。
        return self.aggregate_W(output), graph_attention # 通过聚合线性层处理拼接后的输出，并返回最终的输出和计算得到的注意力权重。
