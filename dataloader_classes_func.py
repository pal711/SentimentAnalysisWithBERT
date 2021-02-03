from bs4 import BeautifulSoup
import torch
from torch.utils.data import Dataset, DataLoader


class ReviewDataset(Dataset):
    
    def __init__(self, reviews, labels, tokenizer, max_len):
        self.reviews= reviews
        self.labels= labels
        self.tokenizer= tokenizer
        self.max_len= max_len
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, item):
        review= self.reviews[item]
        review= BeautifulSoup(review, "html.parser").get_text()
        label= self.labels[item]
        
        encodings= self.tokenizer(
                review,
                padding= 'max_length',
                max_length= self.max_len,
                truncation= True,
                return_tensors= 'pt'
                )
        
        return {
            #'review': review,
            'encoding': encodings,
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    
def createDataLoader(dataset, tokenizer, max_len= 150, batch_size= 32, num_workers=0):
    '''
    dataset is a dictionary with 2 keys 'data', 'labels'
    '''
    encoded_dataset= ReviewDataset(dataset['data'], dataset['labels'], tokenizer, max_len)
    
    return DataLoader(encoded_dataset, batch_size= batch_size, num_workers= num_workers)