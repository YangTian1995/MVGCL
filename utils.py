import torch
import os
import random
import numpy as np


def create_directory(d):
    if d and not os.path.exists(d):
        os.makedirs(d)
    return d


def set_seed(args):
    seed = int(args.seed) # 将命令行参数 args.seed 转换为整数，并将其赋值给变量 seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # 设置 PyTorch 的 CPU 随机种子、GPU 随机种子以及多 GPU 的种子为 seed。
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # 设置 PyTorch 的 cuDNN 库为确定性模式（torch.backends.cudnn.deterministic = True），这确保了在相同输入情况下，每次运行都会产生相同的输出。
    torch.backends.cudnn.benchmark = False
    # 将 cuDNN 的自动调整（benchmarking）关闭（torch.backends.cudnn.benchmark = False），这确保了反向传播的计算速度相对稳定，而不受输入大小的影响。
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # 设置环境变量 CUBLAS_WORKSPACE_CONFIG，这是 NVIDIA cuBLAS 库的一个配置参数，可以影响性能。
    torch.use_deterministic_algorithms(True)
    # 设置 PyTorch 使用确定性算法，这包括启用 cuDNN 的确定性卷积算法。


def collate_fn(batch): # 用于 PyTorch 的 DataLoader 中的 collate_fn 函数，用于在自然语言处理（NLP）任务中进行训练或推理时对批量数据进行整理。
    # 通过将序列填充到最大长度，并将数据转换为 PyTorch 张量，标准化了输入批次
    max_len = max([len(f["input_ids"]) for f in batch])
    max_sent = max([len(f["sent_pos"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    sent_pos = [f["sent_pos"] for f in batch]
    sent_labels = [f["sent_labels"] for f in batch if "sent_labels" in f]
    attns = [f["attns"] for f in batch if "attns" in f]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)

    labels = [torch.tensor(label) for label in labels]
    labels = torch.cat(labels, dim=0)

    if sent_labels != [] and None not in sent_labels:
        sent_labels_tensor = []
        for sent_label in sent_labels:
            sent_label = np.array(sent_label)
            sent_labels_tensor.append(np.pad(sent_label, ((0, 0), (0, max_sent - sent_label.shape[1]))))
        sent_labels_tensor = torch.from_numpy(np.concatenate(sent_labels_tensor, axis=0))
    else:
        sent_labels_tensor = None

    if attns:
        attns = [np.pad(attn, ((0, 0), (0, max_len - attn.shape[1]))) for attn in attns]
        attns = torch.from_numpy(np.concatenate(attns, axis=0))
    else:
        attns = None

    graph = [f["graph"] for f in batch]
    pos_graph = [f["pos_graph"] for f in batch]
    output = (input_ids, input_mask, labels, entity_pos, hts, sent_pos, sent_labels_tensor, attns, graph, pos_graph)

    return output
