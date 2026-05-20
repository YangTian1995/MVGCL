import torch
import torch.nn.functional as F
import numpy as np


def process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens): #上下文池化
    # Split the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024.
    n, c = input_ids.size()
    start_tokens = torch.tensor(start_tokens).to(input_ids) # cls
    end_tokens = torch.tensor(end_tokens).to(input_ids) # sep
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)
    if c <= 512:
        # if document can fit into the encoder
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
        )
        sequence_outputs = torch.stack(output[-2][-3:], dim=1) # 从模型输出中提取最后三层的隐藏状态，将这些状态在新的维度（dim=1）上堆叠起来。
        sequence_output = sequence_outputs.mean(dim=1) # 计算堆叠后的隐藏状态在堆叠维度上的均值，以获得一个对最后三层隐藏状态的平均表示。
        attentions = torch.stack(output[-1][-3:],dim=1) # 类似地，从模型输出中提取最后三层的注意力权重，并在新的维度上堆叠。 多头注意力
        attention = attentions.mean(dim=1)
        
    else:
        new_input_ids, new_attention_mask, num_seg = [], [], []
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist() # 表示的是8个 batch 各自的句子长度，处理完需要重新把这8个batch的输入pad到一样的长度 所以需要这个数字
        for i, l_i in enumerate(seq_len): # for each batch
            if l_i <= 512:
                new_input_ids.append(input_ids[i, :512])
                new_attention_mask.append(attention_mask[i, :512])
                num_seg.append(1) # 对于每个批次中的序列，如果序列长度小于等于512，就直接使用该序列和对应的注意力掩码，标记段数为1。
            else: # split the input into two parts: (0, 512) and (end - 512, end)
                input_ids1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1) # 它将两个张量沿着特定维度拼接起来
                input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)
                # 在基于Transformer的模型中，注意力掩码通常用0和1的值来表示：1 表示模型应该关注该位置的元素。0 表示该位置的元素是填充的，模型在计算注意力时应该忽略这些位置。
                attention_mask1 = attention_mask[i, :512]
                attention_mask2 = attention_mask[i, (l_i - 512): l_i]
                new_input_ids.extend([input_ids1, input_ids2])
                new_attention_mask.extend([attention_mask1, attention_mask2])
                num_seg.append(2) # 并将这些新创建的序列和掩码添加到相应的列表中，段数标记为2。
                
        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)
        
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
        )
            
        sequence_outputs = torch.stack(output[-2][-3:], dim=1)
        sequence_output = sequence_outputs.mean(dim=1)
        attentions = torch.stack(output[-1][-3:],dim=1)
        attention = attentions.mean(dim=1)

        i = 0
        new_output, new_attention = [], []
        for (n_s, l_i) in zip(num_seg, seq_len):
            if n_s == 1: # 1 segment (no split) 如果n_s == 1，表示该序列没有被分割，可以直接处理。使用F.pad函数对序列输出和注意力权重进行填充，以补齐到模型的最大长度（c - 512）
                output = F.pad(sequence_output[i], (0, 0, 0, c - 512)) # F.pad函数用于在张量的各个维度上添加填充，其参数定义了在每个维度上填充的大小。
                att = F.pad(attention[i], (0, c - 512, 0, c - 512))
                new_output.append(output)
                new_attention.append(att)
            elif n_s == 2: # 2 segments (splitted)
                
                # first half
                output1 = sequence_output[i][:512 - len_end]
                mask1 = attention_mask[i][:512 - len_end]
                att1 = attention[i][:, :512 - len_end, :512 - len_end]
                # pad to reserve space for the second half
                output1 = F.pad(output1, (0, 0, 0, c - 512 + len_end))
                mask1 = F.pad(mask1, (0, c - 512 + len_end))
                att1 = F.pad(att1, (0, c - 512 + len_end, 0, c - 512 + len_end))

                # second half
                output2 = sequence_output[i + 1][len_start:]
                mask2 = attention_mask[i + 1][len_start:]
                att2 = attention[i + 1][:, len_start:, len_start:]
                # pad to reserve space for the first half
                output2 = F.pad(output2, (0, 0, l_i - 512 + len_start, c - l_i))
                mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                att2 = F.pad(att2, [l_i - 512 + len_start, c - l_i, l_i - 512 + len_start, c - l_i])
                
                # combine first half and second half 
                mask = mask1 + mask2 + 1e-10 # 并加上一个非常小的数（1e-10）以避免除以零的情况。这个操作保证了在后续的操作中，每个位置至少有一个非零的权重，这对于避免数值问题很重要。
                output = (output1 + output2) / mask.unsqueeze(-1) # mask.unsqueeze(-1)的作用是确保mask的维度与output1和output2相匹配，使得除法可以在逐元素的基础上进行。
                att = (att1 + att2)
                att = att / (att.sum(-1, keepdim=True) + 1e-10) # att.sum(-1, keepdim=True)计算每个注意力权重矩阵在最后一个维度（通常是序列长度）上的和，以保持权重的和为1，+ 1e-10同样是为了避免除以零。归一化确保了在合并后，注意力分布仍然是有效的，并且总和为1。
                new_output.append(output)
                new_attention.append(att)
            i += n_s
            
        sequence_output = torch.stack(new_output, dim=0)
        attention = torch.stack(new_attention, dim=0)

    return sequence_output, attention
