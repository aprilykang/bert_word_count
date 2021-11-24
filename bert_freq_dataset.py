import torch
from torch.utils.data import Dataset



class TextClassificationCollator():

    def __init__(self, tokenizer, max_length, vocab_freq, with_text=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_text = with_text
        self.vocabfreq = vocab_freq

    def __call__(self, samples):
        
        tokenized_text=[self.tokenizer.tokenizer(s['text']) for s in samples]
        #input_ids=[self.tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]
        #freqlist=[self.vocabfreq[s] for s in tokenized_text]
        texts = [s['text'] for s in samples]
        labels = [s['label'] for s in samples]
        #print(self.tokenizer.tokenize(texts))
        
        
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        

        for index, seq in enumerate(tokenized_text):
            temp = [int(self.vocabfreq[i]) for i in seq]
            while(len(temp) < 510):
                temp.append(0)
            #mask=encoding['attention_mask'][index]
            encoding['attention_mask'][index][1:-1]=torch.tensor(temp[0:510])

       # print(len(mask))
        return_value = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long),
        }
        if self.with_text:
            return_value['text'] = texts

        return return_value


class TextClassificationDataset(Dataset):

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        return {
            'text': text,
            'label': label,
        }
