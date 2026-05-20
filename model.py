import torch
import torch.nn as nn
from opt_einsum import contract
import torch.nn.functional as F
from long_seq import process_long_input
from losses import ATLoss
from graph import AttentionGCNLayer,AttentionGCNLayer_2


class DocREModel(nn.Module):

    def __init__(self, args, config, model, tokenizer,
                 emb_size=768, block_size=64, num_labels=-1,
                 max_sent_num=25, evi_thresh=0.2): # max_sent_num 是最大句子数量。evi_thresh 是用于确定证据的阈值。emb_size 是用于提取头部和尾部表示的线性层的输出维度。
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = config.hidden_size

        self.loss_fnt = ATLoss() # 使用自定义的ATLoss类初始化一个损失函数
        self.loss_fnt_evi = nn.KLDivLoss(reduction="batchmean") # 使用PyTorch内置的 KL 散度损失函数 (nn.KLDivLoss) 初始化证据损失函数。并设置了 reduction="batchmean"，这意味着在计算损失时将对每个批次的损失进行平均。

        self.head_extractor = nn.Linear(self.hidden_size * 2, emb_size) # 这两个线性层用于提取头部和尾部的表示。输入维度是两倍的隐藏层大小，输出维度是 emb_size。
        self.tail_extractor = nn.Linear(self.hidden_size * 2, emb_size)

        self.use_graph = args.use_graph
        if self.use_graph:
            self.head_extractor = nn.Linear(3 * config.hidden_size, emb_size)
            self.tail_extractor = nn.Linear(3 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        self.total_labels = config.num_labels
        self.max_sent_num = max_sent_num
        self.evi_thresh = evi_thresh

        self.edges = ['self-loop','mention-anaphor', 'co-reference', 'inter-entity'] # 定义了模型中使用的不同类型的边。这里包括自环（self-loop）、实体提及与照应关系（mention-anaphor）、共指关系（co-reference）和实体之间关系（inter-entity）。
        
        self.triplet_margin = args.triplet_margin if hasattr(args, "triplet_margin") else 0.5  # 支持从参数配置，默认0.5
        self.triplet_loss_fn = nn.TripletMarginLoss(
            margin=self.triplet_margin,  # 边际值，控制正负例距离间隔
            p=2,  # 使用欧氏距离（p=2），也可设为1使用曼哈顿距离
            reduction="mean"  # 对批次内损失取平均
        )
        
        if self.use_graph:
            self.graph_layers = nn.ModuleList(
                AttentionGCNLayer(self.edges, self.hidden_size, nhead=args.attn_heads, iters=args.gcn_layers) for _ in
                range(args.iters)) # 如果使用图，创建了一个由多个图卷积网络层组成的模块列表（nn.ModuleList）。每个图卷积网络层都由 AttentionGCNLayer 类构建，它接受一些参数，包括边的类型、隐藏层的大小、注意力头数和图卷积层的迭代次数。这里通过列表推导式创建了多个相同类型的图卷积网络层。
            self.graph_layers_p = nn.ModuleList(
                AttentionGCNLayer_2(self.edges, self.hidden_size, nhead=args.attn_heads, iters=args.gcn_layers) for _ in
                range(args.iters))
    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        # process long documents.
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)

        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts, offset): # 用于处理语言模型输出和注意力权重，以获取实体对的表示
        n, h, _, c = attention.size() # 获取注意力张量的尺寸信息，n代表批次大小，h代表注意力头数，c代表序列长度。
        hss, tss, rss = [], [], []
        ht_atts = [] # 初始化用于存储头实体表示、尾实体表示和上下文表示的空列表，以及用于存储头尾对的注意力权重的空列表。

        for i in range(len(entity_pos)):  # for each batch
            entity_embs, entity_atts = [], []

            # obtain entity embedding from mention embeddings.
            for eid, e in enumerate(entity_pos[i]):  # for each entity
                if len(e) > 1: # 如果实体包含多个提及。
                    e_emb, e_att = [], []
                    for mid, (start, end) in enumerate(e):  # for every mention
                        if start + offset < c: # 如果提及的起始位置加上偏移量小于序列长度。
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset]) # 获取提及的序列输出和注意力权重，并添加到对应列表中。

                    if len(e_emb) > 0: # 如果提及列表不为空，则计算提及的表示和注意力的均值；否则，使用零张量填充。
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else: # 如果实体只包含一个提及，则直接获取对应位置的序列输出和注意力权重；如果超出序列长度，则使用零张量填充。
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)

                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d] 将实体表示和实体注意力权重转换为张量。
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)

            # obtain subject/object (head/tail) embeddings from entity embeddings.
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0]) # 从实体表示和实体注意力权重中获取头实体、尾实体、头实体注意力和尾实体注意力。
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
 
            ht_att = (h_att * t_att).mean(1)  # average over all heads 计算头尾对注意力的均值，并进行归一化处理，然后添加到列表中。
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-30)
            ht_atts.append(ht_att)

            # obtain local context embeddings.
            rs = contract("ld,rl->rd", sequence_output[i], ht_att) # 计算局部上下文表示。

            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
            
        # 计算每个批次的实体对数量，并将头实体、尾实体、局部上下文表示和头尾对注意力合并为一个张量。
        rels_per_batch = [len(b) for b in hss] 
        hss = torch.cat(hss, dim=0)  # (num_ent_pairs_all_batches, emb_size)
        tss = torch.cat(tss, dim=0)  # (num_ent_pairs_all_batches, emb_size)
        rss = torch.cat(rss, dim=0)  # (num_ent_pairs_all_batches, emb_size)
        ht_atts = torch.cat(ht_atts, dim=0)  # (num_ent_pairs_all_batches, max_doc_len)

        return hss, rss, tss, ht_atts, rels_per_batch

    def graph(self, sequence_output, graphs, attention, entity_pos, hts, offset, graph_layers):
        n, h, _, c = attention.size()

        max_node = max([graph.shape[0] for graph in graphs]) # 这行代码计算了graphs中所有图形中节点数的最大值，并将其赋值给max_node。
        graph_fea = torch.zeros(n, max_node, self.config.hidden_size, device=sequence_output.device) # 这两行创建了两个张量graph_fea和graph_adj，用于存储图形特征和邻接矩阵。
        graph_adj = torch.zeros(n, max_node, max_node, device=sequence_output.device) 

        for i, graph in enumerate(graphs): # 这段代码遍历graphs列表中的每个图形，并将每个图形的邻接矩阵转换为张量后存储在graph_adj中。
            nodes_num = graph.shape[0]
            graph_adj[i, :nodes_num, :nodes_num] = torch.from_numpy(graph)

        for i in range(len(entity_pos)): # 这部分代码根据实体位置信息将模型输出中的特征存储到graph_fea中。如果实体位置加上偏移量超过了c（c是attention张量的维度之一），则使用全零向量填充。
            mention_index = 0
            for e in entity_pos[i]:
                for start, end in e:
                    if start + offset < c:
                        # In case the entity mention is truncated due to limited max seq length.
                        graph_fea[i, mention_index, :] = sequence_output[i, start + offset]
                    else:
                        graph_fea[i, mention_index, :] = torch.zeros(self.config.hidden_size).to(sequence_output)
                    mention_index += 1

        for graph_layer in graph_layers:
            graph_fea, _ = graph_layer(graph_fea, graph_adj) # 这段代码遍历了self.graph_layers中的每一层，对图形特征进行处理。

        h_entity, t_entity = [], [] # 这里初始化了两个空列表，用于存储头实体和尾实体的特征。
        for i in range(len(entity_pos)): # 这部分代码根据实体位置信息从graph_fea中提取实体特征，并将头实体和尾实体的特征分别存储在h_entity和t_entity中。
            entity_embs = []
            mention_index = 0
            for e in entity_pos[i]:
                e_emb = graph_fea[i, mention_index:mention_index + len(e), :]
                mention_index += len(e)

                e_emb = torch.logsumexp(e_emb, dim=0) if len(e) > 1 else e_emb.squeeze(0)
                entity_embs.append(e_emb)

            entity_embs = torch.stack(entity_embs, dim=0)
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
            h_entity.append(hs)
            t_entity.append(ts)

        # 这里将h_entity和t_entity中的特征拼接成一个张量。
        h_entity = torch.cat(h_entity, dim=0)
        t_entity = torch.cat(t_entity, dim=0)
        return h_entity, t_entity

    def forward_rel(self, hs, ts, rs, h, t): # 这是一个模型类的方法，用于进行带有图结构的关系预测的前向传播。它接受头实体表示（hs）、尾实体表示（ts）、上下文表示（rs）、头实体图表示（h）和尾实体图表示（t）作为输入参数。
        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs, h], dim=-1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs, t], dim=-1)))
        # split into several groups.
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)

        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)
        #print(f"rel:bl 的形状: {bl.shape}, 数据类型: {bl.dtype}")
        #print(f"rel:logits 的形状: {logits.shape}, 数据类型: {logits.dtype}")

        return bl,logits

    def forward_rel_no_graph(self, hs, ts, rs):
        # 将头实体表示和关系表示以及尾实体表示和关系表示进行拼接，然后通过两个全连接层（self.head_extractor 和 self.tail_extractor）进行处理，并经过激活函数 tanh 处理。
        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=-1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=-1)))
        # split into several groups.将处理后的头实体和尾实体表示分割成几个组，每个组的形状为（batch_size, emb_size // block_size, block_size）。
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)

        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size) # 通过将头实体和尾实体表示进行张量乘法，然后将结果展平成形状为（batch_size, emb_size * block_size）的张量。  
        logits = self.bilinear(bl) # 通过双线性层（self.bilinear）对展平后的张量进行线性变换，得到最终的预测 logits。
 
        return bl,logits

    def forward_evi(self, doc_attn, sent_pos, batch_rel, offset):
        max_sent_num = max([len(sent) for sent in sent_pos])
        rel_sent_attn = []
        for i in range(len(sent_pos)):  # for each batch
            # the relation ids corresponds to document in batch i is [sum(batch_rel[:i]), sum(batch_rel[:i+1]))
            curr_attn = doc_attn[sum(batch_rel[:i]):sum(batch_rel[:i + 1])]
            curr_sent_pos = [torch.arange(s[0], s[1]).to(curr_attn.device) + offset for s in sent_pos[i]]  # + offset

            curr_attn_per_sent = [curr_attn.index_select(-1, sent) for sent in curr_sent_pos]
            curr_attn_per_sent += [torch.zeros_like(curr_attn_per_sent[0])] * (max_sent_num - len(curr_attn_per_sent))
            sum_attn = torch.stack([attn.sum(dim=-1) for attn in curr_attn_per_sent],
                                   dim=-1)  # sum across those attentions
            rel_sent_attn.append(sum_attn)

        s_attn = torch.cat(rel_sent_attn, dim=0)
        return s_attn

    def find_hard_negatives(self, bl, labels):
        num_ent_pairs = bl.shape[0]
        triplets = []
        device = bl.device 
        for anchor_idx in range(num_ent_pairs):
            # ---------------------- 1. 寻找正例（标签完全相同） ----------------------
            anchor_label = labels[anchor_idx]
            #print(anchor_label)
            positive_mask = torch.all(labels == anchor_label, dim=1)  # 多标签完全匹配
            positive_idx = torch.where(positive_mask)[0]
            #print(len(labels),len(positive_idx))
            # 跳过无有效正例或仅有自身的情况
            if positive_idx.numel() < 2 :
                continue  # 至少需要一个不同的正例
        
            # ---------------------- 2. 逐行计算负例相似度（避免全矩阵） ----------------------
            # 锚点特征 [1, D]
            global_indices = torch.arange(num_ent_pairs, device=device)  # 关键修改：指定设备
            other_mask = global_indices != anchor_idx  # 与bl同设备的掩码
            # 所有样本特征 [N, D]，排除自身
            other_feats = bl[other_mask]
            other_labels = labels[other_mask]
            N_other = other_feats.shape[0]
            
            # 计算与所有其他样本的余弦相似度 [1, N_other]
            anchor_feat = bl[anchor_idx].unsqueeze(0)
            similarity = F.cosine_similarity(anchor_feat, other_feats, dim=-1).squeeze(0)
            
            # ---------------------- 3. 筛选标签不同的负例 ----------------------
            negative_mask = ~torch.all(other_labels == anchor_label, dim=1)  # 标签不同的样本
            valid_neg_indices = torch.where(negative_mask)[0]  # 有效负例在other中的索引
            
            if valid_neg_indices.numel() == 0:
                continue  # 无有效负例，跳过
            
            # 获取有效负例的相似度和原始索引（需转换为全局索引）
            valid_neg_similarity = similarity[valid_neg_indices]
            #valid_neg_global_idx = torch.arange(num_ent_pairs)[torch.arange(num_ent_pairs) != anchor_idx][valid_neg_indices]
            valid_neg_global_idx = global_indices[other_mask][valid_neg_indices]
            # 按相似度降序排序，取前5个候选
            top_neg_indices = valid_neg_global_idx[torch.argsort(valid_neg_similarity, descending=True)[:5]]
            
            # ---------------------- 4. 生成三元组（锚点+第一个正例+最优负例） ----------------------
            # 正例选择：排除自身后的第一个正例（positive_idx 已确保至少有一个不同）
            positive_candidates = positive_idx[positive_idx != anchor_idx]
            positive_rand = torch.randperm(positive_candidates.size(0))
            positive_candidates = positive_candidates[positive_rand]
            #if positive_candidates.numel() == 0:
            #    continue  # 正例不能是自身
            positive_idx_selected = positive_candidates[0]  # 取第一个不同的正例
            #print('pos')
            #print(positive_idx_selected)
            # 负例选择：取top1有效负例
            for neg_idx in top_neg_indices:
                if neg_idx != anchor_idx:  # 双重确保排除自身
                    triplets.append((anchor_idx, positive_idx_selected, neg_idx))
                    break  # 每个锚点取一个最优负例
                    
        return triplets
    
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,  # relation labels
                entity_pos=None,
                hts=None,  # entity pairs
                sent_pos=None,
                sent_labels=None,  # evidence labels (0/1)
                teacher_attns=None,  # evidence distribution from teacher model
                graph=None,
                pos_graph=None,
                tag="train"
                ):

        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        output = {}
        sequence_output, attention = self.encode(input_ids, attention_mask)

        hs, rs, ts, doc_attn, batch_rel = self.get_hrt(sequence_output, attention, entity_pos, hts, offset) # 根据编码后的序列输出、注意力张量、实体位置、实体对信息和偏移量，获取头实体（hs）、关系（rs）、尾实体（ts）、文档级别注意力（doc_attn）和批次级别关系信息（batch_rel）。

        if self.use_graph: # 如果使用图结构（self.use_graph=True），则通过图模型（self.graph）得到实体表示（h和t），然后计算 logits；否则直接计算 logits。
            #h, t = self.graph(sequence_output, graph, attention, entity_pos, hts, offset)
            h, t = self.graph(sequence_output, graph, attention, entity_pos, hts, offset,self.graph_layers)
            #neg_h, neg_t = self.graph(sequence_output, neg_graph, attention, entity_pos, hts, offset,self.graph_layers)
            pos_h, pos_t = self.graph(sequence_output, pos_graph, attention, entity_pos, hts, offset,self.graph_layers_p)
            bl,logits = self.forward_rel(hs, ts, rs, h, t)
            bl_p,logits_p = self.forward_rel(hs, ts, rs, pos_h, pos_t)
            
            
        else:
            bl,logits = self.forward_rel_no_graph(hs, ts, rs)
            
        #logits = 
        output["rel_pred"] = self.loss_fnt.get_label(logits, num_labels=self.num_labels) # 使用损失函数（self.loss_fnt）获取关系预测值，并存储在输出字典中。

        if sent_labels is not None:  # human-annotated evidence available 如果提供了句子标签（sent_labels），表示有人工标注的证据可用。

            s_attn = self.forward_evi(doc_attn, sent_pos, batch_rel, offset) # 通过证据模型（self.forward_evi）计算证据预测值（s_attn），然后将其填充到指定形状后存储在输出字典中。
            output["evi_pred"] = F.pad(s_attn > self.evi_thresh, (0, self.max_sent_num - s_attn.shape[-1]))

        if tag in ["test", "dev"]:  # testing 如果在测试或开发阶段，使用损失函数（self.loss_fnt）获取得分，并存储在输出字典中。这里是因为模型计算出来概率 验证集和测试集 直接取结果就行
            # 训练集需要算损失值，去更新模型，测试集验证集只需要预测的概率去计算指标

            scores_topk = self.loss_fnt.get_score(logits, self.num_labels)
            output["scores"] = scores_topk[0]
            output["topks"] = scores_topk[1]

        if tag == "infer":  # teacher model inference 如果是推理阶段，将文档级别注意力根据批次级别关系信息拆分，并存储在输出字典中。
            output["attns"] = doc_attn.split(batch_rel)

        else:  # training 如果是训练阶段，使用损失函数（self.loss_fnt）计算关系抽取损失，并存储在输出字典中。
            # relation extraction loss
            loss = self.loss_fnt(logits.float(), labels.float())
            output["loss"] = {"rel_loss": loss.to(sequence_output)} # 这部分代码计算了关系抽取的损失。logits是模型的预测结果，labels是真实标签。self.loss_fnt是损失函数，用于计算模型预测值与真实值之间的差异。计算得到的损失存储在output字典中的"loss"字段下，用字典形式表示，其中键为"rel_loss"，对应的值为计算得到的损失。

            # ================== 新增三重损失逻辑 ==================
            if self.use_graph and self.training:  # 仅在使用图结构且训练时启用
                #print(labels)
                # 筛选有效实体对（标签至少有一个正类，与证据损失逻辑一致）
                #valid_mask = labels.sum(dim=-1) > 0  # 多标签场景：总标签和>0
                valid_bl = bl#[valid_mask]
                valid_labels = labels#[valid_mask]
                #print(len(labels))
                #print(len(valid_labels) == len(labels))
                if valid_bl.shape[0] < 3:  # 至少需要1个锚点+1正例+1负例
                    print("Warning: Not enough valid pairs for triplet loss")
                    pass
                else:
                    triplets = self.find_hard_negatives(valid_bl, valid_labels)
                    if triplets:
                        anchor_batch, positive_batch, negative_batch = zip(*triplets)
                        #print(f"valid_bl shape: {valid_bl.shape}")
                        #print(f"anchor_batch shape: {torch.tensor(anchor_batch).shape if isinstance(anchor_batch, tuple) else anchor_batch.shape}")
                        #print(f"anchor_batch type: {type(anchor_batch)}")
                        anchor_batch = torch.tensor(anchor_batch, dtype=torch.long)
                        positive_batch = torch.tensor(positive_batch, dtype=torch.long)
                        negative_batch = torch.tensor(negative_batch, dtype=torch.long)
                        anchor_feats = valid_bl[anchor_batch]
                        positive_feats = bl_p[anchor_batch]
                        negative_feats = valid_bl[negative_batch]
                        
                        # 计算三重损失（使用PyTorch内置TripletMarginLoss）
                        triplet_loss = self.triplet_loss_fn(anchor_feats, positive_feats, negative_feats)
                        output["loss"]["triplet_loss"] = triplet_loss.to(sequence_output)
                        loss += triplet_loss   # 与原有损失联合优化
            
            
            if sent_labels is not None:  # supervised training with human evidence 如果提供了句子标签，表示使用人工标注的证据进行监督训练。

                idx_used = torch.nonzero(labels[:, 1:].sum(dim=-1)).view(-1) # 这部分代码根据标签的情况筛选出有效的数据，将只包含有效数据的部分存储在s_attn和sent_labels中。
                # evidence retrieval loss (kldiv loss)
                s_attn = s_attn[idx_used]
                sent_labels = sent_labels[idx_used]
                norm_s_labels = sent_labels / (sent_labels.sum(dim=-1, keepdim=True) + 1e-30) # 这段代码对句子标签进行归一化处理，确保计算KLDiv损失时不会出现除零错误。
                norm_s_labels[norm_s_labels == 0] = 1e-30
                s_attn[s_attn == 0] = 1e-30
                evi_loss = self.loss_fnt_evi(s_attn.log(), norm_s_labels) # 这里计算了证据检索的损失，使用KLDiv损失函数。计算得到的损失存储在output字典中的"loss"字段下，键为"evi_loss"
                output["loss"]["evi_loss"] = evi_loss.to(sequence_output)

            elif teacher_attns is not None:  # self training with teacher attention 这是另一个条件语句，判断是否提供了教师注意力分布，用于自监督训练。

                doc_attn[doc_attn == 0] = 1e-30 # 这部分代码对教师注意力分布进行处理，避免出现零概率的情况。
                teacher_attns[teacher_attns == 0] = 1e-30
                attn_loss = self.loss_fnt_evi(doc_attn.log(), teacher_attns) # 这里计算了注意力损失，使用KLDiv损失函数。计算得到的损失存储在output字典中的"loss"字段下，键为"attn_loss"。
                output["loss"]["attn_loss"] = attn_loss.to(sequence_output)

        return output
