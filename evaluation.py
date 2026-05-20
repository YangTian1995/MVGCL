import os
import os.path
import json
import numpy as np

rel2id = json.load(open('meta/rel2id.json', 'r'))
id2rel = {value: key for key, value in rel2id.items()}
def get_f1(std,submission_answer,title2vectexSet):
    # 初始化评估指标的变量
    correct_re = 0
    correct_evidence = 0
    pred_evi = 0

    correct_in_train_annotated = 0
    correct_in_train_distant = 0
    titleset2 = set([])
    tot_relations = len(std)
    # 遍历预测结果，计算评估指标
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]
        

        if (title, r, h_idx, t_idx) in std: # 如果预测结果中的关系在标准答案中
            correct_re += 1
            in_train_annotated = in_train_distant = False

            if in_train_annotated: # 如果关系在注释的训练集中出现
                correct_in_train_annotated += 1
            if in_train_distant: # 如果关系在远程监督的训练集中出现
                correct_in_train_distant += 1

    # 计算准确率、召回率和 F1 值
    re_p = 1.0 * correct_re / len(submission_answer) if len(submission_answer) != 0 else 0
    re_r = 1.0 * correct_re / tot_relations if tot_relations != 0 else 0
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)
    return re_p,re_r,re_f1

def get_title2pred(pred: list) -> dict: # 将预测结果转换为字典形式。
    '''
    Convert predictions into dictionary.
    Input:
        :pred: list of dictionaries, each dictionary entry is a predicted relation triple. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score']
    Output:
        :title2pred: dictionary with (key, value) = (title, {rel_triple: score})
    '''

    title2pred = {}

    for p in pred:
        if p["r"] == "Na":
            continue
        curr = (p["h_idx"], p["t_idx"], p["r"])

        if p["title"] in title2pred:
            if curr in title2pred[p["title"]]:
                title2pred[p["title"]][curr] = max(p["score"], title2pred[p["title"]][curr])
            else:
                title2pred[p["title"]][curr] = p["score"]
        else:
            title2pred[p["title"]] = {curr: p["score"]}
    return title2pred

 
def get_title2gt(features: dict) -> dict: # 将标签转换为字典形式的函数。
    '''
    Convert ground-truth labels to dictionary.
    Input:
        :features: list of features within each document. Identical to the lists obtained from pre-processing.
    Output:
        :title2gt: dictionary with (key, value) = (title, [gold_triples])
    '''
    title2gt = {}
    for f in features:
        title = f["title"]
        title2gt[title] = []
        for idx, p in enumerate(f["hts"]):
            h, t = p
            label = np.array(f['labels'][idx])
            rs = np.nonzero(label[1:])[0] + 1  # + 1 for no-label
            title2gt[title].extend([(h, t, id2rel[r]) for r in rs])

    return title2gt


def select_thresh(cand: list, num_gt: int, correct: int, num_pred: int):
    '''
    select threshold for relation predictions.
    Input:
        :cand: list of relation candidates
        :num_gt: number of ground-truth relations.
        :correct: number of correct relation predictions selected.
        :num_pred: number of relation predictions selected.
    Output:
        :thresh: threshold for selecting relations.
        :sorted_pred: predictions selected from cand.
    '''

    sorted_pred = sorted(cand, key=lambda x: x[1], reverse=True)
    precs, recalls = [], []

    for pred in sorted_pred:
        correct += pred[0]
        num_pred += 1
        precs.append(correct / num_pred)  # Precision
        recalls.append(correct / num_gt)  # Recall                             

    recalls = np.asarray(recalls, dtype='float32')
    precs = np.asarray(precs, dtype='float32')
    f1_arr = (2 * recalls * precs / (recalls + precs + 1e-20))
    f1 = f1_arr.max()
    f1_pos = f1_arr.argmax()
    thresh = sorted_pred[f1_pos][1]

    print('Best thresh', thresh, '\tbest F1', f1)
    return thresh, sorted_pred[:f1_pos + 1]


def merge_results(pred: list, pred_pseudo: list, features: list, thresh: float = None):
    '''
    Merge relation predictions from the original document and psuedo documents.
    Input:
        :pred: list of dictionaries, each dictionary entry is a predicted relation triple from the original document. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score'].
        :pred_pseudo: list of dictionaries, each dictionary entry is a predicted relation triple from pseudo documents. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score'].
        :features: list of features within each document. Identical to the lists obtained from pre-processing.
        :thresh: threshold for selecting predictions.
    Output:
        :merged_res: list of merged relation predictions. Each relation prediction is a dictionay with keys (title, h_idx, t_idx, r).
        :thresh: threshold of selecting relation predictions.
    '''

    # 将原始文档和伪文档的预测结果按照标题整理到字典中
    title2pred = get_title2pred(pred)
    title2pred_pseudo = get_title2pred(pred_pseudo)

    # 获取每个文档的真实标签
    title2gt = get_title2gt(features)
    num_gt = sum([len(title2gt[t]) for t in title2gt])

    titles = list(title2pred.keys())
    cand = [] # 候选关系预测列表
    merged_res = []
    correct, num_pred = 0, 0 # 正确预测的数量和总预测数量

    # 遍历每个标题
    for t in titles:
        rels = title2pred[t]
        rels_pseudo = title2pred_pseudo[t] if t in title2pred_pseudo else {}

        union = set(rels.keys()) | set(rels_pseudo.keys())  # 合并原始文档和伪文档的关系预测结果键集合
        for r in union:
            if r in rels and r in rels_pseudo:  # add those into predictions 如果关系在原始文档和伪文档中均有预测
                if rels[r] > 0 and rels_pseudo[r] > 0:  # 如果两者都有正预测
                    merged_res.append({'title': t, 'h_idx': r[0], 't_idx': r[1], 'r': r[2]})  # 如果两者都有正预测
                    num_pred += 1 # 更新总预测数量
                    correct += r in title2gt[t] # 如果预测正确，更新正确预测数量
                    continue
                score = rels[r] + rels_pseudo[r] # 否则，将分数设为两者的分数之和
            elif r in rels:  # -10 for penalty 如果只在原始文档中有预测
                score = rels[r] - 10 # 对于原始文档的预测，减去一个惩罚值
            elif r in rels_pseudo: # 如果只在伪文档中有预测
                score = rels_pseudo[r] - 10 # 对于伪文档的预测，减去一个惩罚值
            cand.append((r in title2gt[t], score, t, r[0], r[1], r[2]))

    # 如果设置了阈值，则根据阈值选择预测结果
    if thresh != None:
        sorted_pred = sorted(cand, key=lambda x: x[1], reverse=True) # 按分数降序排序
        last = min(filter(lambda x: x[1] > thresh, sorted_pred)) # 找到分数大于阈值的最小值
        until = sorted_pred.index(last) # 找到该最小值的索引
        cand = sorted_pred[:until + 1] # 选择分数大于阈值的部分作为候选结果
        merged_res.extend([{'title': r[2], 'h_idx': r[3], 't_idx': r[4], 'r': r[5]} for r in cand])
        return merged_res, thresh

    # 如果候选列表不为空，则根据候选列表的统计信息选择阈值
    if cand != []:
        thresh, cand = select_thresh(cand, num_gt, correct, num_pred)
        merged_res.extend([{'title': r[2], 'h_idx': r[3], 't_idx': r[4], 'r': r[5]} for r in cand])

    return merged_res, thresh


def extract_relative_score(scores: list, topks: list) -> list:
    '''
    Get relative score from topk predictions.
    Input:
        :scores: a list containing scores of topk predictions.
        :topks: a list containing relation labels of topk predictions.
    Output:
        :scores: a list containing relative scores of topk predictions. 包含 topk 预测的相对得分列表
    '''

    # 获取 NA 类别的得分，如果没有 NA 类别，将其设为最后一个得分减去 1
    na_score = scores[-1].item() - 1
    if 0 in topks: # 如果 NA 类别在 topk 中
        na_score = scores[np.where(topks == 0)].item() # # 获取 NA 类别的得分

    # 将所有得分减去 NA 类别的得分，得到相对得分
    scores -= na_score

    return scores


def to_official(preds: list, features: list, evi_preds: list = [], scores: list = [], topks: list = []):
    '''
    Convert the predictions to official format for evaluating.
    Input:
        :preds: list of dictionaries, each dictionary entry is a predicted relation triple from the original document. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score'].
        :features: list of features within each document. Identical to the lists obtained from pre-processing.
        :evi_preds: list of the evidence prediction corresponding to each relation triple prediction.
        :scores: list of scores of topk relation labels for each entity pair.
        :topks: list of topk relation labels for each entity pair.
    Output:
        :official_res: official results used for evaluation.
        :res: topk results to be dumped into file, which can be further used during fushion.
    '''
    h_idx, t_idx, title, sents, pair_types = [], [], [], [], []

    for f in features:
        if "entity_map" in f:
            hts = [[f["entity_map"][ht[0]], f["entity_map"][ht[1]]] for ht in f["hts"]]
        else:
            hts = f["hts"]

        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [f["title"] for ht in hts]
        sents += [len(f["sent_pos"])] * len(hts)
        pair_types += f['pair_types']
    assert len(pair_types) == len(h_idx)
    official_res = []
    res = []

    for i in range(preds.shape[0]):  # for each entity pair
        pred = preds[i]
        #if scores != []:
        if scores.size>0 :
            score = extract_relative_score(scores[i], topks[i]) # # 提取每个实体对的关系得分
            #pred = topks[i] # # 提取每个实体对的前 k 个关系预测结果
        '''
        else:
            pred = preds[i] # # 提取每个实体对的关系预测结果
            pred = np.nonzero(pred)[0].tolist() # 将预测结果中值为 0 的元素排除
        '''
        index = np.where(pred == 1)[0][0] # 找到预测结果中第一个值为 0 的索引，这通常是表示无关的类别
        curr_result = {
                'title': title[i],
                'h_idx': h_idx[i],
                't_idx': t_idx[i],
                'r': id2rel[index],
                'pair_types': pair_types[i]
            }
        if evi_preds != []:
            curr_evi = evi_preds[i]
            evis = np.nonzero(curr_evi)[0].tolist()
            curr_result["evidence"] = [evi for evi in evis if evi < sents[i]]
        #if index != 0:
            #if scores != []:
        if scores.size>0 :
            curr_result["score"] = score[index].item()
        official_res.append(curr_result)
        res.append(curr_result)
        '''
        for p in pred:  # for each predicted relation label (topk)
            curr_result = {
                'title': title[i],
                'h_idx': h_idx[i],
                't_idx': t_idx[i],
                'r': id2rel[p],
            }
            if evi_preds != []:
                curr_evi = evi_preds[i]
                evis = np.nonzero(curr_evi)[0].tolist()
                curr_result["evidence"] = [evi for evi in evis if evi < sents[i]]
            if scores != []:
                curr_result["score"] = score[np.where(topks[i] == p)].item()
            
                official_res.append(curr_result)
            res.append(curr_result)
        '''
    return official_res, res


def gen_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train_"):] # 从数据文件名中提取出包含 "train_" 的部分，表示这是训练集数据文件
    # 将文件名中的 ".json" 替换为 ".fact"，以构建对应的事实文件名
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    # 检查事实文件是否已经存在
    if os.path.exists(fact_file_name):
        # 如果事实文件已存在，则从文件中加载已知的训练集事实并返回
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    # 如果事实文件不存在，则需要从原始数据中提取训练集事实并保存到文件中
    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            # 获取标签对应的关系类型
            rel = label['r']
            # 获取标签中头尾实体对应的节点列表
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    # 将每对实体和关系组成的三元组添加到训练集事实集合中
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train


def official_evaluate(tmp, path, train_file="train_annotated.json", dev_file="dev.json"): # 评估模型在验证集或测试集上的表现，并返回四个不同情形下的评估结果。
    '''
        Adapted from the official evaluation code
    '''
    # 生成标准答案的文件夹路径
    entity_type_to_id = {'GeneOrGeneProduct': 'G', 'DiseaseOrPhenotypicFeature': 'D', 
                                  'SequenceVariant': 'V', 'ChemicalEntity': 'C',
                                    'OrganismTaxon': 'O', 'CellLine': 'CL'}
    truth_dir = os.path.join(path, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    # 获取训练集注释和训练集远程监督中的事实
    fact_in_train_annotated = gen_train_facts(os.path.join(path, train_file), truth_dir)
    #fact_in_train_distant = gen_train_facts(os.path.join(path, "train_distant.json"), truth_dir)

    # 读取验证集或测试集的标准答案
    truth = json.load(open(os.path.join(path, dev_file)))

    pair_set = {'GD','DG','GG','GC','CG','DV','VD','CD','DC','CV','VC','CC','VV'}
    std = {}
    GD_std = {}
    GG_std = {}
    GC_std = {}
    DV_std = {}
    CD_std = {}
    CV_std = {}
    CC_std = {}
    VV_std = {}
    tot_evidences = 0
    titleset = set([])

    title2vectexSet = {}

    # 构建标准答案的字典
    for x in truth:
        title = x['title']
        titleset.add(title)
        vertexSet = x['vertexSet']

        title2vectexSet[title] = vertexSet

        if 'labels' not in x:  # official test set from DocRED
            continue

        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            h_type = entity_type_to_id[vertexSet[h_idx][0]['type']]
            t_type = entity_type_to_id[vertexSet[t_idx][0]['type']]
            p_type = h_type + t_type
    
            if h_idx > t_idx:
                temp = h_idx
                h_idx = t_idx
                t_idx = temp
            #print(label['evidence'])
            if p_type == 'GD' or p_type =='DG':
                GD_std[(title, r, h_idx, t_idx)] = set(label['evidence'])
               
            if p_type == 'GG':
                GG_std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            if p_type == 'GC' or p_type == 'CG':
                GC_std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            if p_type == 'DV' or p_type == 'VD':
                DV_std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            if p_type == 'CD' or p_type == 'DC':
                CD_std[(title, r, h_idx, t_idx)] = set(label['evidence']) 
            if p_type == 'CV' or p_type == 'VC':
                CV_std[(title, r, h_idx, t_idx)] = set(label['evidence']) 
            if p_type == 'CC' or p_type == 'CC':
                CC_std[(title, r, h_idx, t_idx)] = set(label['evidence']) 
            if p_type == 'VV' or p_type == 'VV':
                VV_std[(title, r, h_idx, t_idx)] = set(label['evidence']) 
            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            tot_evidences += len(label['evidence'])

    tot_relations = len(std)
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    
    submission_answer = []
    if tmp[0]['pair_types'] in pair_set:
        submission_answer.append(tmp[0])
    # 对预测结果进行排序和去重
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            if tmp[i]['pair_types'] in pair_set: 
                submission_answer.append(tmp[i])

    GD_answer = []
    GG_answer = []
    GC_answer = []
    DV_answer = []
    CD_answer = []
    CV_answer = []
    CC_answer = []
    VV_answer = []

    for answer in submission_answer:
        p_type = answer['pair_types']  
        if p_type  == 'GD' or p_type =='DG':
            GD_answer.append(answer)
        if p_type == 'GG':
            GG_answer.append(answer)
        if p_type == 'GC' or p_type == 'CG':
            GC_answer.append(answer)
        if p_type == 'DV' or p_type == 'VD':
            DV_answer.append(answer)
        if p_type == 'CD' or p_type == 'DC':
            CD_answer.append(answer)
        if p_type == 'CV' or p_type == 'VC':
            CV_answer.append(answer)
        if p_type == 'CC' or p_type == 'CC':
            CC_answer.append(answer)
        if p_type == 'VV' or p_type == 'VV':
            VV_answer.append(answer)

    GD_p,GD_r,GD_f1 = get_f1(GD_std,GD_answer,title2vectexSet)
    print("GD_p:{:.2f},GD_r:{:.2f},GD_f1:{:.2f}".format(GD_p *100,GD_r*100,GD_f1*100))
    GG_p,GG_r,GG_f1 = get_f1(GG_std,GG_answer,title2vectexSet)
    print("GG_p:{:.2f},GG_r:{:.2f},GG_f1:{:.2f}".format(GG_p*100,GG_r*100,GG_f1*100))
    GC_p,GC_r,GC_f1 = get_f1(GC_std,GC_answer,title2vectexSet)
    print("GC_p:{:.2f},GC_r:{:.2f},GC_f1:{:.2f}".format(GC_p*100,GC_r*100,GC_f1*100))
    DV_p,DV_r,DV_f1 = get_f1(DV_std,DV_answer,title2vectexSet)
    print("DV_p:{:.2f},DV_r:{:.2f},DV_f1:{:.2f}".format(DV_p*100,DV_r*100,DV_f1*100))
    CD_p,CD_r,CD_f1 = get_f1(CD_std,CD_answer,title2vectexSet)
    print("CD_p:{:.2f},CD_r:{:.2f},CD_f1:{:.2f}".format(CD_p*100,CD_r*100,CD_f1*100))
    CV_p,CV_r,CV_f1 = get_f1(CV_std,CV_answer,title2vectexSet)
    print("CV_p:{:.2f},CV_r:{:.2f},CV_f1:{:.2f}".format(CV_p*100,CV_r*100,CV_f1*100))
    CC_p,CC_r,CC_f1 = get_f1(CC_std,CC_answer,title2vectexSet)
    print("CC_p:{:.2f},CC_r:{:.2f},CC_f1:{:.2f}".format(CC_p*100,CC_r*100,CC_f1*100))
    # 初始化评估指标的变量
    correct_re = 0
    correct_evidence = 0
    pred_evi = 0

    correct_in_train_annotated = 0
    correct_in_train_distant = 0
    titleset2 = set([])
    
    # 遍历预测结果，计算评估指标
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if 'evidence' in x:  # and (title, h_idx, t_idx) in std:
            evi = set(x['evidence'])
        else:
            evi = set([])
        pred_evi += len(evi)

        if (title, r, h_idx, t_idx) in std: # 如果预测结果中的关系在标准答案中
            correct_re += 1
            stdevi = std[(title, r, h_idx, t_idx)]
            correct_evidence += len(stdevi & evi)
            in_train_annotated = in_train_distant = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated: # 如果在注释的训练集中出现
                        in_train_annotated = True
                    #if (n1['name'], n2['name'], r) in fact_in_train_distant: # 如果在远程监督的训练集中出现
                    #    in_train_distant = True

            if in_train_annotated: # 如果关系在注释的训练集中出现
                correct_in_train_annotated += 1
            #if in_train_distant: # 如果关系在远程监督的训练集中出现
            #    correct_in_train_distant += 1

    # 计算准确率、召回率和 F1 值
    re_p = 1.0 * correct_re / len(submission_answer)
    re_r = 1.0 * correct_re / tot_relations if tot_relations != 0 else 0
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

    evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = 1.0 * correct_evidence / tot_evidences if tot_evidences > 0 else 0

    if evi_p + evi_r == 0:
        evi_f1 = 0
    else:
        evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

    # 计算忽略训练集注释和忽略整个训练集时的指标
    re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (
            len(submission_answer) - correct_in_train_annotated + 1e-5)
    re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (
            len(submission_answer) - correct_in_train_distant + 1e-5)

    if re_p_ignore_train_annotated + re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

    if re_p_ignore_train + re_r == 0:
        re_f1_ignore_train = 0
    else:
        re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

    return [re_p, re_r, re_f1], [evi_p, evi_r, evi_f1], \
        [re_p_ignore_train_annotated, re_r, re_f1_ignore_train_annotated], \
        [re_p_ignore_train, re_r, re_f1_ignore_train]
