import pickle
import json
import numpy as np

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.id2word = {}
        self.idx = 0
        self.add_word('<pad>')
        self.add_word('<end>')
        self.add_word('<start>')
        self.add_word('<unk>')

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.id2word[self.idx] = word
            self.idx += 1

    def get_word_by_id(self, id):
        return self.id2word[id]

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

with open('vocab.pkl','rb') as f:
    vocab = pickle.load(f)

with open('findings_dict.json') as f:
    data = json.load(f)

keys = list(data.keys())

max_len = 0

for key in keys:
    item = data[key]
    item = item.split(' ')
    if len(item) > max_len:
        max_len = len(item)
# print(max_len)

max_len+=2
print(max_len)
for key in keys:
    name = key.split('.')[0]
    item = data[key]
    item = item.split(' ')
    item.insert(0,'<start>')
    item.append('<end>')
    while len(item) != max_len:
        item.append('<pad>')
    new_data = np.zeros((max_len,),dtype=np.int)
    for i in range(max_len):
        new_data[i] = vocab(item[i])
    np.save('/mnt/Data2/Preethi/XRay_Report_Generation/Data/text_data_xml_findings/'+name,new_data)

# with open('/mnt/Data2/Preethi/XRay_Report_Generation/Data/img_dict.json') as f:
#     img_data = json.load(f)

# for key in keys:
#     item = data[key]
#     item = item.split(' ')
#     item.insert(0,'<start>')
#     item.append('<end>')
#     while len(item) != max_len:
#         item.append('<pad>')
#     new_data = np.zeros((max_len,),dtype=np.int)
#     for i in range(max_len):
#         new_data[i] = vocab(item[i])
    
#     img_names = img_data[key]
#     if len(img_names) != 0:
#         for name in img_names:
#             np.save('/mnt/Data2/Preethi/XRay_Report_Generation/Data/text_data_img_impressions/'+name,new_data)            