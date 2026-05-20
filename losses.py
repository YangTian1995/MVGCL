import torch
import torch.nn as nn
import torch.nn.functional as F


class ATLoss(nn.Module):
    '''
    def forward(self, logits, labels): # 接受两个参数，logits 是模型的输出，labels 是真实标签。
        # TH label 
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels) # 创建一个与输入标签 labels 形状相同的全零张量，数据类型为浮点数。to(labels) 将张量移到与 labels 相同的设备上（CPU或GPU）。
        th_label[:, 0] = 1.0 # 将第一个类别（索引为0）设为正类别。将 th_label 中第一列的所有元素设为1.0。
        labels[:, 0] = 0.0 # 将输入标签 labels 中的第一列（第一个类别）的所有元素设为0.0。这是因为在计算损失时，要将原始标签中的第一个类别的概率置零，以便对其他类别进行排序。

        p_mask = labels + th_label # p_mask 是一个掩码，其目的是在计算损失函数时仅考虑正类别。
        n_mask = 1 - labels # n_mask 是另一个掩码，其目的是在计算损失函数时仅考虑负类别。labels 是原始标签，1 - labels 将原始标签中的非零位置变为零，将零位置变为1，从而排除正类别。
        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30 # logit1 即是将模型输出中正类别位置的得分调整为极小值的结果。
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1) # 这样得到的 loss1 是排名正类别的损失项。
        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)
        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss
    '''
    
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
    
        #print(logits.shape,labels.shape)
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)

        return loss

    '''
    def get_label(self, logits, num_labels=-1): # 它接受两个参数：logits（模型的输出）和 num_labels（要返回的标签数量，默认值为 -1）。

        th_logit = logits[:, 0].unsqueeze(1)  # theshold is norelation。这行代码提取 logits 的第一列（假设第一列代表“无关系”类别），并在其上添加一个新的维度。
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit) # 创建一个掩码，表示 logits 中大于“无关系”阈值的元素。
        if num_labels > 0: # 如果 num_labels 大于0，意味着用户希望获取得分最高的前 num_labels 个标签。
            top_v, _ = torch.topk(logits, num_labels, dim=1) # 使用 torch.topk 获取每行得分最高的前 num_labels 个元素。
            top_v = top_v[:, -1] # smallest logits among the num_labels。从 top_v 中提取最小的元素。
            # predictions are those logits > thresh and logits >= smallest
            mask = (logits >= top_v.unsqueeze(1)) & mask # 更新掩码，包括得分在最高 num_labels 之内且大于“无关系”阈值的元素。
        output[mask] = 1.0 # 将满足条件的位置在 output 中设置为1。
        # if no such relation label exist: set its label to 'Nolabel'
        output[:, 0] = (output.sum(1) == 0.).to(logits) # 如果某行在 output 中没有任何标签被选中，则将“无关系”类别设为1。
        return output 
    '''
    def get_label(self, logits, num_labels=-1): # 此函数接受两个参数：logits（模型的输出）和 num_labels（可选，返回的标签数量，默认为 -1）。
        output = torch.zeros_like(logits).to(logits) # 创建一个与 logits 同形状的全零张量。
            
        if num_labels > 0:  # 只关心得分最高的标签
            _, max_indices = torch.max(logits, dim=1)  # 在每一行中找到得分最高的索引。
            output.scatter_(1, max_indices.unsqueeze(1), 1.0)  # 使用 scatter_ 将输出张量中对应的位置设为 1。
        else:
            raise ValueError("num_labels must be 1 for selecting the highest score label only.")

        return output
 
        
    def get_score(self, logits, num_labels=-1):

        if num_labels > 0:
            #print(num_labels)
            return torch.topk(logits, num_labels, dim=1)
        else:
            return logits[:,1] - logits[:,0], 0 # 如果num_labels为0或负数，返回logits张量中前两个类别分数的差异，以及值0。
