import argparse
import os # 导入 Python 的 os 模块，提供了访问操作系统功能的接口。
import numpy as np
import torch # 导入 PyTorch，一个用于深度学习的开源深度学习库。
import ujson as json
from torch.cuda.amp import GradScaler # 从 PyTorch 中的 torch.cuda.amp 模块导入 GradScaler 类，该类用于在混合精度训练中缩放梯度。
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer # 从 Hugging Face 的 transformers 库中导入 AutoConfig、AutoModel 和 AutoTokenizer，这些是用于加载预训练模型的工具。
#from transformers.optimization import AdamW, get_linear_schedule_with_warmup # 从 transformers 库中导入优化器相关的类和函数，包括 AdamW 优化器学习率调度器。
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from args import add_args
from model import DocREModel # 从 model 文件中导入 DocREModel 文档关系抽取的模型。
from utils import set_seed, collate_fn, create_directory # 从 utils 文件中导入一些功能，包括设置随机种子、数据批处理函数和创建目录的功能。
from prepro import read_docred # 从 prepro 文件中导入 read_docred 函数，用于预处理文档关系抽取任务的数据。
from evaluation import to_official, official_evaluate, merge_results
from tqdm import tqdm # 导入 tqdm 库，用于在循环中显示进度条。
import pandas as pd # 导入 pandas 库，用于数据分析和处理。
import pickle # 导入 Python 的 pickle 模块，用于序列化和反序列化对象。
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print("testlog")
def load_input(batch, device, tag="dev"):
    input = {'input_ids': batch[0].to(device),
             'attention_mask': batch[1].to(device),
             'labels': batch[2].to(device).squeeze(),
             'entity_pos': batch[3],
             'hts': batch[4],
             'sent_pos': batch[5],
             'sent_labels': batch[6].to(device) if (not batch[6] is None) and (batch[7] is None) else None,
             'teacher_attns': batch[7].to(device) if not batch[7] is None else None,
             'graph': batch[8],
             'pos_graph': batch[9],
             'tag': tag
             }
    
    return input


def train(args, model, train_features, dev_features): # 定义了一个名为 train 的函数，接受训练相关参数、模型、训练特征和验证特征作为输入。
    def finetune(features, optimizer, num_epoch, num_steps): # 在 train 函数内部定义了一个名为 finetune 的内部函数，用于微调模型。该函数接受特征、优化器、训练轮数和当前步数作为输入。
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn,
                                      drop_last=True) # 创建一个训练数据加载器，用于从特征中加载批次数据进行训练。批次大小由参数指定，并进行了打乱和填充处理。
        train_iterator = range(int(num_epoch)) # 创建一个迭代器，用于迭代训练轮数。
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps) # 计算总步数，考虑到梯度累积的影响。
        warmup_steps = int(total_steps * args.warmup_ratio) # 计算学习率预热步数。
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps) # 使用线性学习率预热器创建调度器，用于动态调整学习率。
        scaler = GradScaler() # 创建一个梯度缩放器，用于在混合精度训练中缩放损失和梯度
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        for epoch in train_iterator: # 迭代训练轮数，并在 tqdm 进度条中显示当前训练轮数。
            print("EPHOCH: {}".format(epoch))
            for step, batch in enumerate(tqdm(train_dataloader)): # 迭代训练数据批次。
            #for step, batch in enumerate(train_dataloader):
                model.zero_grad() # 清空模型和优化器的梯度，并将模型设置为训练模式。
                optimizer.zero_grad()
                model.train()
                inputs = load_input(batch, args.device) # 加载输入数据并将标签设为None，然后通过模型前向传播计算输出。
                #print(f"labels_shapes: {inputs['labels'].shape}")
                inputs["sent_labels"] = None
                outputs = model(**inputs) # 这行代码使用了 Python 的可变关键字参数（keyword arguments）传递方式，即 **inputs。这意味着 inputs 是一个字典，其中包含了模型的输入参数及其对应的数值。
                loss = [outputs["loss"]["rel_loss"]] # 计算损失，将关系损失添加到损失列表中。

                if inputs["sent_labels"] is not None: # 如果输入中存在句子标签（sent_labels不是空值），则将评估损失（evi_loss）与 args.evi_lambda 相乘后加入损失列表中。
                    loss.append(outputs["loss"]["evi_loss"] * args.evi_lambda)
 
                if inputs["teacher_attns"] is not None: # 如果输入中存在教师注意力分布（teacher_attns不是空值），则将注意力损失（attn_loss）与 args.attn_lambda 相乘后加入损失列表中。
                    loss.append(outputs["loss"]["attn_loss"] * args.attn_lambda)

                loss = sum(loss) / args.gradient_accumulation_steps # 将损失列表中的所有损失相加并除以梯度累积步数（gradient_accumulation_steps）得到平均损失。
                scaler.scale(loss).backward() # 对平均损失进行反向传播，并使用梯度缩放器（scaler）进行梯度缩放。

                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0: # 如果设置了最大梯度范数限制（max_grad_norm大于0），则将梯度缩放器（scaler）取消缩放，并对模型参数的梯度进行裁剪，以确保梯度的范数不超过指定的最大值。
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer) # 通过调用梯度缩放器的 step 方法来更新优化器的参数。然后更新梯度缩放器的状态，并调用学习率调度器的 step 方法来更新学习率。最后，将模型的梯度归零。
                    scaler.update()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1

                # 如果当前步数是训练数据加载器的最后一步，或者已经达到了评估步数（evaluation_steps）并且当前步数是梯度累积步数的整数倍，则执行以下步骤：
                if (step + 1) == len(train_dataloader) or (
                        args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0): 

                    dev_scores, dev_output, official_results, results = evaluate(args, model, dev_features, tag="dev")
#
                    print(dev_output)
                    if dev_scores["dev_F1_ign"] > best_score:
                        best_score = dev_scores["dev_F1_ign"]
                        best_offi_results = official_results
                        best_results = results
                        best_output = dev_output

                        ckpt_file = os.path.join(args.save_path, "best.ckpt")
                        print(f"saving model checkpoint into {ckpt_file} ...")
                        torch.save(model.state_dict(), ckpt_file)

                    if epoch == train_iterator[-1]:  # last epoch 如果当前轮次是训练迭代器的最后一轮，则保存最后的模型参数到文件，并保存预测结果到文件。

                        ckpt_file = os.path.join(args.save_path, "last.ckpt")
                        print(f"saving model checkpoint into {ckpt_file} ...")
                        torch.save(model.state_dict(), ckpt_file)

                        pred_file = os.path.join(args.save_path, args.pred_file)
                        score_file = os.path.join(args.save_path, "scores.csv")
                        results_file = os.path.join(args.save_path, f"topk_{args.pred_file}")

                        dump_to_file(best_offi_results, pred_file, best_output, score_file, best_results, results_file)

        return num_steps

    new_layer = ["extractor", "bilinear", "graph"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": args.lr_added},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr_transformer, eps=args.adam_epsilon) # 初始化优化器
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps) # 调用 finetune 函数进行微调训练。


def evaluate(args, model, features, tag="dev"): # 定义了一个名为evaluate的函数，接受四个参数：args是参数配置，model是要评估的模型，features是要用于评估的特征数据，tag表示评估的标签，默认为"dev"。
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn,
                            drop_last=False) # 使用DataLoader加载特征数据，以便按批次进行评估。其中features是要加载的数据，batch_size是批次大小，shuffle表示是否在每个迭代中打乱数据顺序，collate_fn是用于对样本进行处理的函数，drop_last表示是否丢弃最后一个不完整的批次。
    preds, evi_preds = [], [] # 始化两个空列表preds和evi_preds，用于存储模型预测结果。
    scores, topks = [], [] # 初始化两个空列表scores和topks，用于存储评估结果中的得分和Top-K值。
    attns = [] # 初始化空列表attns，用于存储注意力权重。

    for batch in tqdm(dataloader):
        model.eval() # 将模型设置为评估模式，这会影响一些层（如Dropout和BatchNormalization），使其在评估时表现更好。

        if args.save_attn:
            tag = "infer"

        inputs = load_input(batch, args.device, tag) # 调用load_input函数，将当前批次数据加载到设备上，并根据标签进行处理。

        with torch.no_grad(): # 使用torch.no_grad()上下文管理器，表示在此范围内的操作不会计算梯度，以节省内存和加快速度。     
            inputs["sent_labels"] = None
            outputs = model(**inputs)
            pred = outputs["rel_pred"]
            pred = pred.cpu().numpy() # 将预测结果移动到CPU上，并转换为NumPy数组。
            pred[np.isnan(pred)] = 0 # 将预测结果中的NaN值（如果有的话）替换为0。
            preds.append(pred)

            if "scores" in outputs:
                scores.append(outputs["scores"].cpu().numpy())
                topks.append(outputs["topks"].cpu().numpy())

            if "evi_pred" in outputs:  # relation extraction and evidence extraction
                evi_pred = outputs["evi_pred"]
                evi_pred = evi_pred.cpu().numpy()
                evi_preds.append(evi_pred)

            if "attns" in outputs:  # attention recorded
                attn = outputs["attns"]
                attns.extend([a.cpu().numpy() for a in attn])

    preds = np.concatenate(preds, axis=0) # 将所有批次的预测结果连接起来，形成一个完整的预测结果。

    if scores:
        scores = np.concatenate(scores, axis=0)
        topks = np.concatenate(topks, axis=0)

    if evi_preds:
        evi_preds = np.concatenate(evi_preds, axis=0)

    official_results, results = to_official(preds, features, evi_preds=evi_preds, scores=scores, topks=topks) # 调用to_official函数，将预测结果转换为官方格式，并计算其他相关指标。

    if len(official_results) > 0: # 检查是否存在官方结果。
        if tag == "test":
            best_re, best_evi, best_re_ign, _ = official_evaluate(official_results, args.data_dir, args.train_file,
                                                                  args.test_file)
        else:
            best_re, best_evi, best_re_ign, _ = official_evaluate(official_results, args.data_dir, args.train_file,
                                                                  args.dev_file)
    else:
        best_re = best_evi = best_re_ign = [-1, -1, -1]
        
    #将评估结果组织成字典output和scores，并返回。
    output = {
        tag + "_rel": [i * 100 for i in best_re],
        tag + "_rel_ign": [i * 100 for i in best_re_ign],
        tag + "_evi": [i * 100 for i in best_evi],
    }
    scores = {"dev_F1": best_re[-1] * 100, "dev_evi_F1": best_evi[-1] * 100, "dev_F1_ign": best_re_ign[-1] * 100}

    if args.save_attn:
        attns_path = os.path.join(args.load_path, f"{os.path.splitext(args.test_file)[0]}.attns")
        print(f"saving attentions into {attns_path} ...")
        with open(attns_path, "wb") as f:
            pickle.dump(attns, f)

    return scores, output, official_results, results


def dump_to_file(offi: list, offi_path: str, scores: list, score_path: str, results: list = [], res_path: str = "",
                 thresh: float = None):
    '''
    dump scores and (top-k) predictions to file.
    
    '''
    # 将官方评估结果保存到文件中
    print(f"saving official predictions into {offi_path} ...")
    json.dump(offi, open(offi_path, "w"))

    print(f"saving evaluations into {score_path} ...")
    headers = ["precision", "recall", "F1"]
    scores_pd = pd.DataFrame.from_dict(scores, orient="index", columns=headers)
    print(f"scores_pd:{scores_pd}")
    scores_pd.to_csv(score_path, sep='\t')

    # 如果存在预测结果，则将预测结果保存到文件中
    if len(results) != 0:
        assert res_path != ""
        print(f"saving topk results into {res_path} ...")
        json.dump(results, open(res_path, "w"))

    if thresh is not None:
        thresh_path = os.path.join(os.path.dirname(offi_path), "thresh")
        if not os.path.exists(thresh_path):
            print(f"saving threshold into {thresh_path} ...")
            json.dump(thresh, open(thresh_path, "w"))

    return


def main():
    parser = argparse.ArgumentParser() # 创建了一个 ArgumentParser 对象 parser，用于解析命令行参数。
    parser = add_args(parser) # 使用 add_args 函数向 parser 添加了一些特定的命令行参数，该函数返回修改后的 parser。
    args = parser.parse_args() # 使用 parser.parse_args() 解析命令行参数，并将结果存储在 args 变量中。

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 检查是否有可用的 CUDA 设备。如果有，将 device 设置为 "cuda:0"，否则设置为 "cpu"。
    args.n_gpu = torch.cuda.device_count() # 使用 torch.cuda.device_count() 获取可用的 CUDA 设备数量，并将结果存储在 args.n_gpu 中。
    args.device = device # 将 device 赋值给 args.device。

    config = AutoConfig.from_pretrained( # 使用 AutoConfig.from_pretrained() 从预训练模型加载配置信息。
        args.config_name if args.config_name else args.model_name_or_path, # 如果提供了 args.config_name，则使用它；否则，使用 args.model_name_or_path。
        num_labels=args.num_class, # Number of relation types in dataset.
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    if args.special_token:
        tokenizer.add_special_tokens({'additional_special_tokens':['\"','G145R','P120T','Aceruloplasminemia','AD',
                                                                            '@/GeneOrGeneProduct','GeneOrGeneProduct/@',
                                                                            '@/DiseaseOrPhenotypicFeature','DiseaseOrPhenotypicFeature/@',
                                                                            '@/SequenceVariant','SequenceVariant/@',
                                                                            '@/ChemicalEntity', 'ChemicalEntity/@',
                                                                            '@/CellLine','CellLine/@'
                                                                            ]})                                                                      
    
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path), # 设置 from_tf 为 True 如果 args.model_name_or_path 包含 ".ckpt"，否则为 False。
        config=config,
    )

    config.transformer_type = args.transformer_type # 将 args.transformer_type 赋值给 config.transformer_type
 
    set_seed(args) # 这个函数的作用是设置随机种子，以确保实验的可重复性。args 是通过命令行传递的参数，用于指定随机种子   

    read = read_docred # 将函数赋值给变量 read。用于读取 DocRED 数据集的函数 read_docred
    
    '''
    这段 JSON 数据看起来描述了一个关于 "Skai TV" 的文档。以下是对 JSON 结构的解释：

    "vertexSet": 一个包含多个元素的列表，每个元素代表一个实体的相关信息。

    每个元素都是一个列表，表示同一类型的实体可能在不同句子中出现的情况。
    每个实体信息包括：
    "pos": 实体在文本中的起始和结束位置。
    "type": 实体类型，如 "ORG"（组织）、"LOC"（地点）、"TIME"（时间）、"MISC"（其他）等。
    "sent_id": 实体所在句子的索引。
    "name": 实体的名称。
    "labels": 一个包含多个元素的列表，每个元素代表文档中的一对关系。

    每个元素包括：
    "r": 关系类型。
    "h": 关系中的头实体在 "vertexSet" 中的索引。
    "t": 关系中的尾实体在 "vertexSet" 中的索引。
    "evidence": 证据列表，包含了支持该关系的句子索引。
    "title": 文档的标题，表示为字符串。

    "sents": 一个包含多个元素的列表，每个元素代表文档中的一个句子。

    每个元素是一个包含多个词汇的列表，表示句子中的单词。
    '''

    config.cls_token_id = tokenizer.cls_token_id # 这两行代码将tokenizer中的特殊token（如[CLS]和[SEP]）的ID赋值给模型配置（config）中的相应字段，以便模型使用。
    config.sep_token_id = tokenizer.sep_token_id

    model = DocREModel(args, config, model, tokenizer,
                       num_labels=args.num_labels,
                       max_sent_num=args.max_sent_num,
                       evi_thresh=args.evi_thresh) # 创建了一个DocREModel的实例，调用forward函数.例如，假设model是一个PyTorch模型的实例，那么通过model(input)调用模型时，实际上是在调用model的forward方法。
    model.to(args.device)
    model.model.resize_token_embeddings(len(tokenizer))
    print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters() if p.requires_grad])) # 这行代码打印出模型中可训练参数的总数量。

    if args.load_path != "":  # load model from existing checkpoint

        model_path = os.path.join(args.load_path, "best.ckpt") # 如果指定了加载路径，则加载预训练好的模型参数。加载的路径通常是之前训练好的模型的保存路径。
        model.load_state_dict(torch.load(model_path))

    if args.do_train:  # Training

        create_directory(args.save_path) # 创建一个目录，用于保存训练过程中生成的模型及其相关文件。

        train_file = os.path.join(args.data_dir, args.train_file) # 获取训练数据文件和验证数据文件的路径。
        dev_file = os.path.join(args.data_dir, args.dev_file)

        train_features = read(train_file, tokenizer, transformer_type=args.transformer_type,
                              max_seq_length=args.max_seq_length, teacher_sig_path=args.teacher_sig_path) # 读取并处理训练数据和验证数据的特征。
        dev_features = read(dev_file, tokenizer, transformer_type=args.transformer_type,
                            max_seq_length=args.max_seq_length)

        train(args, model, train_features, dev_features) # 调用train函数进行模型训练。

    else:  # Testing

        basename = os.path.splitext(args.test_file)[0]
        test_file = os.path.join(args.data_dir, args.test_file)

        test_features = read(test_file, tokenizer, transformer_type=args.transformer_type,
                             max_seq_length=args.max_seq_length)

        if args.eval_mode != "fushion":

            test_scores, test_output, official_results, results = evaluate(args, model, test_features, tag="test")

            offi_path = os.path.join(args.load_path, args.pred_file)
            score_path = os.path.join(args.load_path, f"{basename}_scores.csv")
            res_path = os.path.join(args.load_path, f"topk_{args.pred_file}")

            dump_to_file(official_results, offi_path, test_output, score_path, results, res_path)

        else:  # inference stage cross fusion 推断阶段的交叉融合
            # 从文件中加载预测结果
            results = json.load(open(os.path.join(args.results_path, f"topk_{args.pred_file}")))

            # formulate pseudo documents from top-k (k=num_labels in arguments) predictions 从top-k（k=参数中的标签数量）预测中构建伪文档
            pseudo_test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length,
                                        single_results=results)

            # 评估伪文档的分数和输出
            pseudo_test_scores, pseudo_output, pseudo_official_results, pseudo_results = evaluate(args, model,
                                                                                                  pseudo_test_features,
                                                                                                  tag="test")

            # 如果在结果路径中存在“thresh”文件，则加载阈值
            if 'thresh' in os.listdir(args.results_path):
                with open(os.path.join(args.results_path, "thresh")) as f:
                    thresh = json.load(f)
                print(f"Threshold loaded from file: {thresh}")
            else:
                thresh = None

            
            # 合并真实结果和伪结果，以及可能的阈值
            merged_offi, thresh = merge_results(results, pseudo_results, test_features, thresh)
            # 官方评估合并后的结果
            merged_re, merged_evi, merged_re_ign, _ = official_evaluate(merged_offi, args.data_dir, args.train_file,
                                                                        args.test_file)
            
            # 设置结果的标签
            tag = args.test_file.split('.')[0]
            merged_output = {
                tag + "_rel": [i * 100 for i in merged_re],
                tag + "_rel_ign": [i * 100 for i in merged_re_ign],
                tag + "_evi": [i * 100 for i in merged_evi],
            }

            
            offi_path = os.path.join(args.results_path, f"fused_{args.pred_file}")
            score_path = os.path.join(args.results_path, f"{basename}_fused_scores.csv")
            dump_to_file(merged_offi, offi_path, merged_output, score_path, thresh=thresh)


if __name__ == "__main__":
    print("test_log")
    main() 