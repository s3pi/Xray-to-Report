import pickle
import json
import numpy as np

special_tokens = ['<fstop>','<alt>','<num>','<unk>','<pos>']

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.id2word = {}
        self.idx = 0
        self.add_word('<pad>')
        self.add_word('<end>')
        self.add_word('<start>')
        self.add_word('<unk>')
        self.add_word('<fstop>')
        self.add_word('<alt>')
        self.add_word('<num>')
        self.add_word('<pos>')

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

vocab = Vocabulary()

with open('caption_dict.json', 'r') as f:
    data = json.load(f)
keys = list(data.keys())
# print(keys)

unique_words = []
count_words = []

for key in keys:
    item = data[key]
    # words = item.split(' ')
    for word in item.split(' '):
        if word not in special_tokens:
            if word in unique_words:
                index = unique_words.index(word)
                count_words[index]+=1.0
            else:
                unique_words.append(word)
                count_words.append(1.0)
# print(len(unique_words))
# print(len(count_words))
count_words = np.asarray(count_words)
total_freq = np.sum(count_words)
count_words = count_words / total_freq
# print(np.sum(count_words))
sorting_index = np.argsort(count_words)
i = sorting_index.shape[0] - 1
sorted_unique_words = []
sorted_count_words = []
while i >= 0:
    sorted_unique_words.append(unique_words[int(sorting_index[i])])
    sorted_count_words.append(count_words[int(sorting_index[i])])
    i = i - 1
# print(len(sorted_unique_words))
# exit()
cum_prob = 0.0
i = 0
# print(vocab.idx)
# for i in range(len(sorted_count_words)):
#     cum_prob = cum_prob+sorted_count_words[i]
# print(cum_prob)
# exit()
while vocab.idx < 1000:
    vocab.add_word(sorted_unique_words[i])
    cum_prob = cum_prob + sorted_count_words[i]
    i+=1
print(cum_prob)
print(vocab.idx)
print('YAAAAY')
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)