import json
import pandas as pd

def process_data_format(original_file, output_file):
      input_data = json.load(original_file)     
      # 初始化输出数据列表
      output_data = []

      # 遍历输入数据中的每个文档
      
      for doc in input_data['documents']:
      # 初始化当前文档的vertexSet、sentences和labels列表
            vertex_set = []
            sentences = []
            labels = []
            entity_mentions = {}
            passage_text = ''
            last_index = 0
            sents_location = []
            for passage_index,passage in enumerate(doc['passages']):
                  # 初始化当前passage的实体列表
                  sent_location = [passage["offset"],passage["offset"] + len(passage['text'])]
                  passage_text += ' ' + passage['text']
                  annotations =passage['annotations']
                  sents_location.append(sent_location)
            passage_text = passage_text.strip(' ')
            for passage_index,passage in enumerate(doc['passages']):
                  sentence = []
                  annotations =passage['annotations']

                  for annotation in annotations:
                        
                        locations = annotation['locations']
                        mention_offset = locations[0]['offset']
                        mention_length = locations[0]['length']
                        if mention_offset > last_index:
                              sentence.append(passage_text[last_index:mention_offset])
                        sentence.append(passage_text[mention_offset:mention_offset+mention_length])
                        last_index = mention_offset+mention_length
                        entity_identifier = annotation["infons"]["identifier"]
                        mention_text = annotation["text"]
                        entity_identifiers = entity_identifier.split(',')
                        for entity_identifier in entity_identifiers:
                              if entity_identifier not in entity_mentions:
                                    entity_mentions[entity_identifier] = []
                              entity_mentions[entity_identifier].append({'text':mention_text,
                                                            'type':annotation["infons"]["type"],
                                                            'locations':[len(sentence) - 1, len(sentence)],
                                                            'sent_id':passage_index})
            
                  if last_index < len(passage_text):
                        sentence.append(passage_text[last_index:])
                  sentences.append(sentence)
            #分句并且获得句子的起止位置     
            
            
            
            sents_location = []
            sent_offset =0
            processed_sents = []
            allword2loc = []
            word_offset = 0
            '''
            for p_id, passage_text in enumerate(sentences):
                  
                  passage_text.replace("    ","88 8")
                  passage_text.replace("   ",'7 8')
                  flag = 0
                  if p_id == 1:
                        flag = 1
                  if flag == 0:
                        passage_text += ' '
                  sentence_split = passage_text.split(". ")                
                  
                  if sentence_split[-1] == '':
                        del sentence_split[-1]
                  for id,sentence in enumerate(sentence_split):
                        word2loc = []
                        processed_sent = []
                        # 这行代码移除了字符串sentence两端的所有空格。函
                        words = sentence.split(" ")
                        # 移除空字符串
                        for word in words:
                              word2loc.append([word_offset ,word_offset + len(word)])
                              word_offset += len(word) + 1 
                        word_offset += 1
                        allword2loc.append(word2loc)
                        words = [word for word in words if word]
                        # 将分割后的单词和标点添加到处理后的句子中
                        processed_sent.extend(words)
                        processed_sent.append('.')
                        # 将处理后的句子添加到结果列表中
                        processed_sents.append(processed_sent)
                        
                        sent_location = []
                        sent_location.append(sent_offset + flag)
                        sent_location.append(sent_offset + len(sentence) + 1 + flag) 
                        sent_offset += len(sentence) + 1 + flag
                        sents_location.append(sent_location)
            '''
            # 初始化当前passage的实体列表        
            #entity_mentions = {}
            # 遍历当前文档中的每个passage
            '''
            for passage in doc['passages']:
                  
                  # 遍历当前passage中的每个annotation（实体）
                  for annotation in passage['annotations']:
                        entity_identifier = annotation["infons"]["identifier"]
                        mention_text = annotation["text"]
                        entity_identifiers = entity_identifier.split(',')
                        for entity_identifier in entity_identifiers:
                              if entity_identifier not in entity_mentions:
                                    entity_mentions[entity_identifier] = []
                              entity_mentions[entity_identifier].append({'text':mention_text,
                                                            'type':annotation["infons"]["type"],
                                                            'locations':[annotation['locations'][0]['offset'], annotation['locations'][0]['offset'] + annotation['locations'][0]['length']],
                                                            'sent_id':0})
            '''

            # 将当前passage的文本添加到sentences列表中    
            for entity_identifier, entity in entity_mentions.items():
                  sentence_id = 0
                  for entity_mention in entity:
                        mention_start = entity_mention['locations'][0]
                        mention_end = entity_mention['locations'][1]
                        # 在句子列表中查找实体提及属于哪个句子
                        '''
                        for i,sentence_location in enumerate(sents_location):
                              sentence_start = sentence_location[0]
                              sentence_end = sentence_location[1]
                              if mention_start >= sentence_start and mention_end <= sentence_end:
                                    # 实体提及属于当前句子
                                    sentence_id = i
                                    entity_mention['sent_id'] = sentence_id
                                    break
                        mention_start_loc = 0
                        mention_end_loc = 0
                        for i,word_location in enumerate(allword2loc[sentence_id]):
                              if mention_start == word_location[0]:
                                    mention_start_loc = i
                                    break
                              #考虑这种情况  EPO/EPOR是word 但是实体是EPOR 
                              if mention_start < word_location[0]:
                                    mention_end_loc = i - 1
                                    break
                        for i,word_location in enumerate(allword2loc[sentence_id]):
                              if mention_end <= word_location[1]:
                                    mention_end_loc = i
                                    entity_mention['locations'] = [mention_start_loc, mention_end_loc + 1]
                                    break
                        assert mention_start_loc != mention_end_loc + 1
                        '''
                        
                  # 完事儿再遍历entity_mentions字典，取出各个信息
                  entity_info = [{
                  "pos": mention['locations'],
                  "type": mention["type"],
                  "sent_id": mention['sent_id'], 
                  "name": mention["text"],
                  "identifier": entity_identifier
                  } for mention in entity]  
                  # 将实体添加到当前passage的实体列表中
                  # 将当前passage的实体列表添加到vertexSet中
                  vertex_set.append(entity_info)

            # 构建当前文档的labels列表
            entity_id = list(entity_mentions.keys())
            for relation in doc['relations']:
                  label = {
                        "r": relation['infons']['type'],
                        "h": entity_id.index(relation['infons']['entity1']),
                        "t": entity_id.index(relation['infons']['entity2']),
                        "evidence": []  # 假设没有提供证据信息
                  }
                  labels.append(label) 
                  
            doc_structure = {
                  "vertexSet": vertex_set,
                  "labels": labels,
                  "title": doc['id'], 
                  "sents": sentences
            }
            
            output_data.append(doc_structure)
      json.dump(output_data,output_file, ensure_ascii=False, indent=4)
      output_file.write('\n')

#with open('dataset/BioRED/Train.BioC.JSON','r') as original_file, \
#with  open('dataset/BioRED/debug_train.json','r') as original_file, \
with open('dataset/BioRED/Train.BioC.JSON','r') as original_file, \
      open('dataset/BioRED_format_new/Train.BioC.JSON','w')  as output_file:
     process_data_format(original_file = original_file,output_file = output_file)
    
with open('dataset/BioRED/Test.BioC.JSON','r') as original_file, \
      open('dataset/BioRED_format_new/Test.BioC.JSON','w')  as output_file:
     process_data_format(original_file = original_file,output_file = output_file)

with open('dataset/BioRED/Dev.BioC.JSON','r') as original_file, \
      open('dataset/BioRED_format_new/Dev.BioC.JSON','w')  as output_file:
     process_data_format(original_file = original_file,output_file = output_file)
     
     