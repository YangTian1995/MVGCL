from tqdm import tqdm
import ujson as json # 导入 ujson 库并重命名为 json，ujson 是一个用于高效处理 JSON 数据的库。
import numpy as np
import pickle # 导入 pickle 模块，用于序列化和反序列化 Python 对象。
import os # 导入 Python 的 os 模块，用于与操作系统交互，例如文件路径等。
import graph_util
entity_type_to_insert_tok = {'GeneOrGeneProduct':('@/GeneOrGeneProduct','GeneOrGeneProduct/@'),
                            'DiseaseOrPhenotypicFeature':('@/DiseaseOrPhenotypicFeature','DiseaseOrPhenotypicFeature/@'),
                            'SequenceVariant':('@/SequenceVariant','SequenceVariant/@'),
                            'ChemicalEntity':('@/ChemicalEntity', 'ChemicalEntity/@'),
                            'CellLine':('@/CellLine','CellLine/@'),
                            'OrganismTaxon':('@/OrganismTaxon','OrganismTaxon/@')}
docred_rel2id = json.load(open('meta/rel2id.json', 'r'))
docred_ent2id = {'NA': 0, 'ORG': 1, 'LOC': 2, 'NUM': 3, 'TIME': 4, 'MISC': 5, 'PER': 6} # 创建一个实体到ID的映射字典，表示了一些预定义的实体类型。

from spacy.tokens import Doc # 从 SpaCy 库的 tokens 模块导入 Doc 类，用于表示文档。
import spacy # 导入 SpaCy 库，这是一个用于自然语言处理的强大工具库。

nlp = spacy.load('en_core_web_sm') # 使用 SpaCy 加载英语模型 'en_core_web_sm'，该模型包含了一系列用于处理英语文本的语言处理工具。


def get_anaphors(sents, mentions): # 但仅限于潜在代词的名称不在已有列表 mentions 中的情况。这样可以确保列表中的代词信息是唯一的，不会重复。如
    potential_mentions = [] # sents = [] # 用于存储处理后的文本

    for sent_id, sent in enumerate(sents): 
        # 确保除了命名实体识别之外的其他处理步骤也被应用于文档。这样，doc_spacy 对象将包含多个语言处理方面的信息，如词性、句法结构等。                            
        doc_spacy = Doc(nlp.vocab, words=sent) # 创建了一个 SpaCy 的 Doc 对象，表示待处理的文档。nlp.vocab 表示使用的词汇表，而 words=sent 则将句子作为单词序列传递给文档。
        for name, tool in nlp.pipeline: # 遍历 SpaCy 处理管道中的每个处理工具。
            if name != 'ner':
                tool(doc_spacy)

        for token in doc_spacy:
            # 初始化一个空字符串，用于存储潜在代词的名称
            potential_mention = ''
            if token.dep_ == 'det' and token.text.lower() == 'the': # 判断当前标记是否为定冠词（det）且为 "the"
                # 这段代码的目的是获取以 "the" 开头的定冠词短语的完整文本内容。token.idx 表示当前标记在文档中的起始位置，token.head.idx 表示当前标记所属的短语（头部）的起始位置，通过这两者可以提取定冠词短语的文本内容。
                potential_name = doc_spacy.text[token.idx:token.head.idx + len(token.head.text)] 
                # 获取定冠词短语在文档中的起始和结束位置。token.i 表示当前标记在句子中的索引，通过将其与短语长度相加，可以得到结束位置。
                pos_start, pos_end = token.i, token.i + len(potential_name.split(' '))
                # 创建一个包含潜在代词信息的字典。其中：
                # 'pos': 包含潜在代词在文档中的起始和结束位置的列表。
                # 'type': 设为 'MISC'，表示潜在代词的类型。
                # 'sent_id': 表示句子的 ID。
                # 'name': 包含潜在代词文本内容的字符串。
                potential_mention = {
                    'pos': [pos_start, pos_end],
                    'type': 'MISC',
                    'sent_id': sent_id,
                    'name': potential_name
                }
            if token.pos_ == 'PRON':
                potential_name = token.text
                pos_start = sent.index(token.text) # 获取代词在当前句子中的起始位置。sent.index(token.text) 返回代词在句子中第一次出现的位置。
                potential_mention = {
                    'pos': [pos_start, pos_start + 1],
                    'type': 'MISC',
                    'sent_id': sent_id,
                    'name': potential_name
                }

            if potential_mention:
                if not any(mention in potential_mention['name'] for mention in mentions): # 如果潜在代词的名称不在已有列表中，执行下面的操作。
                    potential_mentions.append(potential_mention)

    return potential_mentions


def create_graph(entity_pos):
    anaphor_pos, entity_pos = entity_pos[-1], entity_pos[:-1] # 将entity_pos列表中的最后一个元素作为anaphor_pos，将其他元素作为entity_pos。
    mention_num = len([mention for entity in entity_pos for mention in entity]) # 计算实体数量和代词（anaphor）数量。
    anaphor_num = len(anaphor_pos)

    N_nodes = mention_num + anaphor_num
    nodes_adj = np.zeros((N_nodes, N_nodes), dtype=np.int32) # 初始化一个二维数组nodes_adj，表示节点之间的邻接关系，初始值为0。

    edges_cnt = 1
    # add self-loop
    for i in range(N_nodes): # 为每个节点添加自环，表示节点与自身的关系
        nodes_adj[i, i] = edges_cnt

    edges_cnt = 2
    # add mention-anaphor edges
    for i in range(mention_num): # 为实体间和实体与代词之间的关系添加边。
        for j in range(mention_num, N_nodes):
            nodes_adj[i, j] = edges_cnt
            nodes_adj[j, i] = edges_cnt

    entities = []
    i = 0
    for e in entity_pos: # 为每个实体创建一个列表，其中包含属于该实体的节点索引。
        ms = []
        for _ in e:
            ms.append(i)
            i += 1
        entities.append(ms)

    edges_cnt = 3
    # add co-reference edges
    for e in entities: # 为具有相同实体的不同节点之间添加边，表示它们的共指关系。
        if len(e) == 1:
            continue
        for m1 in e:
            for m2 in e:
                if m1 != m2:
                    nodes_adj[m1, m2] = edges_cnt

    edges_cnt = 4
    # add inter-entity edges
    nodes_adj[nodes_adj == 0] = edges_cnt # 为所有剩余的节点之间添加边，表示它们之间的关系。

    return nodes_adj


def add_entity_markers(sample, tokenizer, entity_start, entity_end, entity_type = None):
    ''' add entity marker (*) at the end and beginning of entities. '''

    sents = [] # 用于存储处理后的文本
    sent_map = [] # 存储每个 token 在新文本中的位置映射
    sent_pos = [] # 存储每个句子在新文本中的起始和结束位置

    sent_start = 0
    for i_s, sent in enumerate(sample['sents']):
        # add * marks to the beginning and end of entities
        new_map = {} # 存储 token 在新文本中的位置映射

        for i_t, token in enumerate(sent): # 遍历句子中的每个 token，使用 enumerate 获取 token 的索引 i_t 和 token 内容 token。
            tokens_wordpiece = tokenizer.tokenize(token) # 使用给定的分词器 tokenizer 对 token 进行分词，得到 wordpiece tokens。
            if (i_s, i_t) in entity_start:
                index = entity_start.index( (i_s, i_t))
                mention_type = entity_type[index]
                #if mention_type == "organismTaxon":
                    #continue
                #start_token, end_token = entity_type_to_insert_tok[mention_type]
                tokens_wordpiece = ['*'] + tokens_wordpiece
                #tokens_wordpiece = [start_token] + tokens_wordpiece
            if (i_s, i_t) in entity_end:
                #tokens_wordpiece = tokens_wordpiece + [end_token]
                tokens_wordpiece = tokens_wordpiece + ['*']
            new_map[i_t] = len(sents)
            sents.extend(tokens_wordpiece)

        sent_end = len(sents) # 记录当前句子在新文本中的结束位置。
        # [sent_start, sent_end)
        sent_pos.append((sent_start, sent_end,))
        sent_start = sent_end

        # update the start/end position of each token.
        new_map[i_t + 1] = len(sents) # 更新字典 new_map，记录当前 token 在新文本中的位置。
        sent_map.append(new_map) # 将 new_map 添加到 sent_map 列表中，记录当前句子中每个 token 在新文本中的位置映射。

    return sents, sent_map, sent_pos


def get_pseudo_features(raw_feature: dict, pred_rels: list, entities: list, sent_map: dict, offset: int,
                        tokenizer=None): # 用于构建伪标签的函数
# get_pseudo_features(feature[0], title2preds[sample["title"]], entities, sent_map, offset, tokenizer)
    ''' Construct pseudo documents from predictions.'''
    '''
    raw_feature: 原始特征，是一个字典，包含了模型的输入信息，如 input_ids、entity_pos 等。
    pred_rels: 预测的关系列表，包含了从单次运行的预测中得到的关系。
    entities: 包含实体信息的列表。
    sent_map: 映射关系，将句子中的标记映射到其在整个文档中的位置。
    offset: 偏移量，用于调整一些特征。
    tokenizer: 分词器，用于将文本转换为模型可接受的输入格式。
    '''
    pos_samples = 0
    neg_samples = 0

    sent_grps = [] # 是一个列表，用于存储已创建的句子组，避免重复创建。
    pseudo_features = [] # 是一个列表，用于存储伪标签的特征。

    for pred_rel in pred_rels:
        curr_sents = pred_rel["evidence"]  # evidence sentence, 从当前预测关系中提取证据句子集合。
        if len(curr_sents) == 0: # 如果证据句子集合为空，说明该预测关系没有提供任何证据，直接跳过，不进行后续处理。
            continue
        
        # 对于每一个预测的关系，提取其关联的证据句子（evidence sentence）
        # check if head/tail entity presents in evidence. if not, append sentence containing the first mention of head/tail into curr_sents
        head_sents = sorted([m["sent_id"] for m in entities[pred_rel["h_idx"]]]) # 通过 pred_rel["h_idx"] 获取头实体在 entities 中的索引，然后提取该实体所在的所有句子的 sent_id，最后使用 sorted 进行排序。
        tail_sents = sorted([m["sent_id"] for m in entities[pred_rel["t_idx"]]])

        if len(set(head_sents) & set(curr_sents)) == 0: # 如果头实体的句子没有出现在当前的证据句列表 curr_sents 中，就将头实体的第一个句子（head_sents[0]）加入到 curr_sents 中。
            curr_sents.append(head_sents[0])
        if len(set(tail_sents) & set(curr_sents)) == 0:
            curr_sents.append(tail_sents[0])

        curr_sents = sorted(set(curr_sents)) # 确保 curr_sents 中的句子是唯一的，并按照升序进行排序。
        if curr_sents in sent_grps:  # skip if such sentence group has already been created
            continue
        sent_grps.append(curr_sents) # 检查当前的证据句组是否已经存在，如果已经存在，则跳过（不重复创建）。如果不存在，则将当前的证据句组加入到 sent_grps 列表中。

        # new sentence masks and input ids
        old_sent_pos = [raw_feature["sent_pos"][i] for i in curr_sents] # 获取当前证据句组中每个句子的原始句子位置信息。
        new_input_ids_each = [raw_feature["input_ids"][s[0] + offset:s[1] + offset] for s in old_sent_pos] #  根据原始句子位置信息，从原始输入 ID 列表中截取相应的子序列，得到每个句子的新输入 ID 列表。
        new_input_ids = sum(new_input_ids_each, []) # 将所有新的输入 ID 列表拼接成一个大列表。
        new_input_ids = tokenizer.build_inputs_with_special_tokens(new_input_ids) # 构建新的输入特征，包括添加特殊标记（如 [CLS] 和 [SEP]）。

        new_sent_pos = [] # 初始化前一个句子结束位置。

        prev_len = 0 # 初始化了一个计数器 prev_len，用于跟踪累积的句子长度。
        for sent in old_sent_pos:
            curr_sent_pos = (prev_len, prev_len + sent[1] - sent[0])
            new_sent_pos.append(curr_sent_pos)
            prev_len += sent[1] - sent[0]

        # iterate through all entities, keep only entities with mention in curr_sents.

        # obtain entity positions w.r.t whole document
        curr_entities = []
        ent_new2old = {}  # head/tail of a relation should be selected，一个字典用于映射新实体索引到旧实体索引。
        new_entity_pos = []

        for i, entity in enumerate(entities):
            curr = []
            curr_pos = []
            for mention in entity:
                # 对于输入的每个实体，它遍历实体的每个提及（mention）。如果提及位于当前关系的证据句子中（mention["sent_id"] 在 curr_sents 中），则将该提及添加到 curr 列表和相应的调整后的位置信息 curr_pos 中。
                if mention["sent_id"] in curr_sents:
                    curr.append(mention)
                    prev_len = new_sent_pos[curr_sents.index(mention["sent_id"])][0]
                    pos = [sent_map[mention["sent_id"]][pos] - sent_map[mention["sent_id"]][0] + prev_len for pos in
                           mention['pos']]
                    curr_pos.append(pos)

            if curr: # 如果 curr 列表非空，表示该实体至少有一个提及与当前关系有关，那么将整个 curr 列表（包含有关提及的所有信息）添加到 curr_entities 列表中，并将 curr_pos 列表添加到 new_entity_pos 中。同时，更新 ent_new2old 字典，将新实体索引映射到旧实体索引。
                curr_entities.append(curr)
                new_entity_pos.append(curr_pos)
                ent_new2old[len(ent_new2old)] = i  # update dictionary

        # check if anaphor is in pseudo document
        anaphor_in_pseudo = False
        if not entities[-1]: # 检查最后一个实体是否为空（not entities[-1]）。如果为空，表示伪文档中不存在指代词，于是将空列表添加到 new_entity_pos 和 curr_entities 中。
            new_entity_pos.append([])
            curr_entities.append([])
        else: # 如果最后一个实体不为空，表示伪文档中存在指代词。
            for e in curr_entities[-1]: # 遍历 curr_entities[-1] 中的每个实体提及（e）以及 entities[-1] 中的每个指代词（anaphor）。
                for anaphor in entities[-1]:
                    if e['name'] == anaphor['name']:
                        anaphor_in_pseudo = True
            if anaphor_in_pseudo is False:
                new_entity_pos.append([])
                curr_entities.append([])

        # iterate through all entities to obtain all entity pairs
        new_hts = []
        new_labels = []
        for h in range(len(curr_entities) - 1): # 初始化 new_hts 和 new_labels 列表，用于存储新的头尾实体对和对应的标签。
            for t in range(len(curr_entities) - 1):
                if h != t:
                    new_hts.append([h, t])
                    old_h, old_t = ent_new2old[h], ent_new2old[t] # 通过映射字典 ent_new2old 获取原始头尾实体对的索引 old_h 和 old_t。
                    curr_label = raw_feature["labels"][raw_feature["hts"].index([old_h, old_t])] # 通过在原始标签列表 raw_feature["labels"] 中查找对应关系的标签，将标签添加到 new_labels 中。
                    new_labels.append(curr_label)

                    neg_samples += curr_label[0] # 更新正负样本计数，根据当前标签中的第一个元素（0 或 1）。
                    pos_samples += 1 - curr_label[0]
 
        graph = create_graph(new_entity_pos) # 利用新的实体位置信息创建图结构，用于表示实体之间的关系。

        pseudo_feature = {'input_ids': new_input_ids,
                          'entity_pos': new_entity_pos if new_entity_pos[-1] != [] else new_entity_pos[:-1], #果最后一个元素是空列表，则剔除这个空列表。这是为了避免在 pseudo_feature 中保留一个没有实体的空实体位置信息。
                          'labels': new_labels,
                          'hts': new_hts,
                          'sent_pos': new_sent_pos,
                          'sent_labels': None,
                          'title': raw_feature['title'],
                          'entity_map': ent_new2old,
                          'graph': graph
                          }
        pseudo_features.append(pseudo_feature)

    return pseudo_features, pos_samples, neg_samples


def read_docred(file_in,
                tokenizer,
                transformer_type="bert",
                max_seq_length=1024,
                teacher_sig_path="",
                single_results=None):
    entity_type_to_id = {'GeneOrGeneProduct': 'G', 'DiseaseOrPhenotypicFeature': 'D', 
                                  'SequenceVariant': 'V', 'ChemicalEntity': 'C',
                                    'OrganismTaxon': 'O', 'CellLine': 'CL'}
# 包括文件路径 file_in、分词器 tokenizer、transformer 类型 transformer_type、最大序列长度 max_seq_length、teacher signature 文件路径 teacher_sig_path 和单一结果 single_results。
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
# 初始化一些变量，包括行索引 i_line、正样本数量 pos_samples、负样本数量 neg_samples 和特征列表 features。

    if file_in == "":
        return None

    with open(file_in, "r", encoding='utf-8') as fh:
        data = json.load(fh)

    # 第二步infusion inference的时候 去加载之前保存下来的attention权重
    if teacher_sig_path != "":  # load logits
        basename = os.path.splitext(os.path.basename(file_in))[0] # 通过 os.path.basename(file_in) 获取输入文件路径的基本文件名，然后通过 os.path.splitext() 分离文件名和扩展名，最终获得没有扩展名的文件名。这个文件名将被用于构建教师签名文件的名称。
        attns_file = os.path.join(teacher_sig_path, f"{basename}.attns") # 使用 os.path.join() 构建教师签名文件的完整路径，连接 teacher_sig_path 和构建的文件名，并附加扩展名 ".attns"。
        attns = pickle.load(open(attns_file, 'rb')) # 打开教师签名文件，并使用 pickle.load() 加载文件中的内容。这里使用 'rb' 参数表示以二进制模式读取文件，因为 pickle 通常以二进制格式保存对象。

    if single_results != None:
        # reorder predictions as relations by title
        pred_pos_samples = 0 # 用于记录预测中正样本的数量。
        pred_neg_samples = 0
        pred_rels = single_results # 即预测的关系列表。
        title2preds = {} # 初始化一个空字典 title2preds，用于将预测的关系按标题重新组织。 
        for pred_rel in pred_rels: # 遍历 pred_rels 中的每个元素（每个预测的关系）：
            if pred_rel["title"] in title2preds:
                title2preds[pred_rel["title"]].append(pred_rel) # 如果关系的标题 pred_rel["title"] 已经在 title2preds 中存在，将当前关系追加到对应标题的列表中。
            else:
                title2preds[pred_rel["title"]] = [pred_rel] # 如果关系的标题不在 title2preds 中，创建一个新的标题条目，并将当前关系作为列表的第一个元素。

    #for doc_id in tqdm(range(len(data)), desc="Loading examples"):
    for doc_id in range(len(data)):

        sample = data[doc_id]
        entities = sample['vertexSet'] # 获取当前文档的样本（一个文档的数据），存储在变量 sample 中。
        entity_start, entity_end, entity_type= [], [], []
        # record entities
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))
                entity_type.append(mention['type'])
        # add entity markers
        sents, sent_map, sent_pos = add_entity_markers(sample, tokenizer, entity_start, entity_end, entity_type = entity_type) # 调用 add_entity_markers 函数，该函数是用于在文本中添加实体标记的，其中包括一些特殊标记，以便模型能够识别和处理这些实体。

        # training triples with positive examples (entity pairs with labels)
        train_triple = {} # 初始化一个空字典 train_triple，用于记录训练三元组

        if "labels" in sample: # 将标签中的实体对、关系和证据添加到训练三元组 train_triple 中。
            for label in sample['labels']:
                evidence = label['evidence']
                r = int(docred_rel2id[label['r']])
                # 每个标签包含关系（r）、实体头部（h）、实体尾部（t）和证据（evidence）等信息。
                # update training triples
                if label['h'] == label['t']:
                    continue
                if (label['h'], label['t']) not in train_triple :
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}] # 如果实体对不在 train_triple 中，将该实体对添加到 train_triple，并创建一个包含当前关系和证据的字典列表。
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence}) # 如果实体对已经在 train_triple 中，将当前关系和证据追加到相应实体对的字典列表中。

        # get anaphors in the doc
        mentions = set([m['name'] for e in entities for m in e])

        potential_mention = get_anaphors(sample['sents'], mentions) # 调用一个名为 get_anaphors 的函数，该函数用于获取文本中的指代关系。传递了文档中的句子 (sample['sents']) 和实体名称的集合 (mentions)。

        entities.append(potential_mention)

        # entity start, end position
        entity_pos = []

        for e in entities:
            entity_pos.append([]) # 为当前实体创建一个空列表，用于存储实体中每个词的位置信息。
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]] # 获取提及起始位置在文档中的绝对位置。
                end = sent_map[m["sent_id"]][m["pos"][1]]
                label = m["type"] # 获取实体类型标签，例如 "ORG"（组织）、"LOC"（地点）、"TIME"（时间）等。
                entity_pos[-1].append((start, end,)) # 将当前词汇在文档中的起始和结束位置以及实体类型添加到当前实体的位置列表中。

        relations, hts, sent_labels, pair_types = [], [], [], [] # hts 用于存储头尾实体（head entity）的信息。
        # 用于存储句子标签的信息。这可能表示关系在文本中的哪些句子中存在或发生。

        # train_triple 有每个标签包含关系（r）、实体头部（h）、实体尾部（t）和证据（evidence）等信息。
        for h, t in train_triple.keys(): # for every entity pair with gold relation 
            h_type = entities[h][0]['type'] 
            t_type = entities[t][0]['type'] 
            h_id = entity_type_to_id[h_type]
            t_id = entity_type_to_id[t_type]
            relation = [0] * len(docred_rel2id) # 初始化一个二进制列表relation，长度由关系类型的数量决定（len(docred_rel2id)）
            sent_evi = [0] * len(sent_pos) # 初始化一个列表sent_evi，长度由句子位置的数量决定（len(sent_pos)）。
            
            for mention in train_triple[h, t]:  # for each relation mention with head h and tail t  
                
                #使所有实体对都h < t
                if h > t:
                    temp = h
                    h = t
                    t = temp 
                  
                # 对于每个具有头实体h和尾【实体t的关系提及，通过将与关系类型相对应的索引设置为1，更新relation列表。同时，通过增加在句子中找到关系证据的每个句子索引的计数，更新sent_evi列表。
                relation[mention["relation"]] = 1 
                for i in mention["evidence"]:
                    sent_evi[i] += 1
            pair_types.append(h_id + t_id)
            relations.append(relation)
            hts.append([h, t])
            sent_labels.append(sent_evi)
            pos_samples += 1 # 初始化在外层
        '''
        # 这段代码是在为模型准备负样本（negative samples）。具体而言，它使用两个嵌套的循环遍历所有实体对，并对于那些没有关系的实体对执行以下操作：
        for h in range(len(entities) - 1):
            #if entities[h][0]['type'] == 'OrganismTaxon':
            h_type = entities[h][0]['type']
            h_id = entity_type_to_id[h_type]
                #continue
            for t in range(len(entities) - 1): 
                t_type = entities[t][0]['type'] 
                t_id = entity_type_to_id[t_type]
                #if entities[t][0]['type'] == 'OrganismTaxon':
                    #continue
                # all entity pairs that do not have relation are treated as negative samples
                if h < t and [h, t] not in hts:  # and [t, h] not in hts:
                    relation = [1] + [0] * (len(docred_rel2id) - 1) # 初始化一个二进制列表relation，其中第一个元素为1，表示负样本。其余元素为0，对应于其他关系类型。
                    sent_evi = [0] * len(sent_pos) # 初始化一个列表sent_evi，长度由句子位置的数量决定（len(sent_pos)）。
                    relations.append(relation)
                    pair_types.append(t_id+t_id)
                    hts.append([h, t])
                    sent_labels.append(sent_evi)
                    neg_samples += 1
                '''
        graph = create_graph(entity_pos)  # 使用实体位置信息（entity_pos）创建图谱（graph）。
        g_copy = graph.copy()
        g_copy[g_copy == 4] = 0
        #print(g_copy)
        g_copys = []
        for l in range(1,4):
            g_t = g_copy.copy()
            g_t[g_copy != l] = 0

            pos_graph = graph_util.edge_feature(len(graph), g_t)
            pos_graph[g_t == 0] = 0
            pos_graph = np.nan_to_num(pos_graph, nan=0)
            g_copys.append(pos_graph)
        #print(g_copys)
        pos_graph = sum(g_copys)
        #assert len(relations) == ((len(entities) - 1) * (len(entities) - 2))/2 # 进行断言检查，确保关系列表的长度等于实体对的数量乘以（实体对的数量减2）。
        sents = sents[:max_seq_length - 2]  # truncate, -2 for [CLS] and [SEP]。对句子进行截断，保留前max_seq_length - 2个令牌，其中减去2是为了留出[CLS]和[SEP]的位置。
        input_ids = tokenizer.convert_tokens_to_ids(sents) # 将句子转换为对应的输入ID（input_ids）。
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids) # 使用特殊令牌构建包含[CLS]和[SEP]的输入序列
        if hts == []:
            continue
        feature = [{'input_ids': input_ids,
                    'entity_pos': entity_pos if entity_pos[-1] != [] else entity_pos[:-1],
                    'labels': relations,
                    'hts': hts,
                    'sent_pos': sent_pos,
                    'pair_types': pair_types,
                    'sent_labels':sent_labels,
                    'title': sample['title'],
                    'graph': graph,
                    'pos_graph':pos_graph
                    }]

        # 如果存在教师模型的关注分布 (attns)，则将其添加到特征中。这部分代码假设 attns 是一个包含注意力分布的列表，其中每个元素对应一个文档，然后从中选择当前文档的注意力分布并添加到特征中。
        if teacher_sig_path != '':  # add evidence distributions from the teacher model
            feature[0]['attns'] = attns[doc_id][:, :len(input_ids)] # 这个attens 第一个维度应该是所有样本的个数

        if single_results is not None:  # 是为了根据 是否输入single_result 来判断是否进行 infusion inference
            offset = 1 if transformer_type in ["bert", "roberta"] else 0 #  这行代码根据 transformer_type 的类型设置 offset。在一些Transformer模型（如BERT和RoBERTa）中，第一个token是特殊的[CLS] token，因此在处理时可能需要做一些偏移。
            if sample["title"] in title2preds:
                # 如果标题在 title2preds 中存在匹配的预测，就调用 get_pseudo_features 函数生成伪文档的特征。这个函数会使用预测的信息构建新的特征，用于后续的模型训练。
                feature, pos_sample, neg_sample, = get_pseudo_features(feature[0], title2preds[sample["title"]],
                                                                       entities, sent_map, offset, tokenizer)
                pred_pos_samples += pos_sample
                pred_neg_samples += neg_sample

        i_line += len(feature) # 将生成的伪文档特征的数量添加到总行数计数 (i_line) 中
        features.extend(feature) # 这个过程的目的可能是通过集成来自不同预测结果的信息，提高模型的性能和鲁棒性。

    print("# of documents {}.".format(i_line))
    if single_results is not None:
        print("# of positive examples {}.".format(pred_pos_samples))
        print("# of negative examples {}.".format(pred_neg_samples))

    else:
        print("# of positive examples {}.".format(pos_samples))
        print("# of negative examples {}.".format(neg_samples))

    return features
