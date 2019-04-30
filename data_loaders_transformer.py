from torchtext.data import Field, Dataset, Iterator, Example
import torch
import os

def train_iterator(device, embedding_dim=200, batch_size=16, train=True):
    d = os.path.dirname(__file__)

    SENTENCE = Field(sequential=True, lower=True, include_lengths=False, batch_first=True, init_token='<sos>', eos_token='<eos>')        
    train_ds = _load_dataset(SENTENCE, path=os.path.join(d, 'training.txt'))
    SENTENCE.build_vocab(train_ds, vectors="glove.6B.{}d".format(embedding_dim), specials=['<sos>'])

    return (Iterator(train_ds, batch_size=batch_size, train=train, device=device), SENTENCE.vocab)
    
def valid_iterator(device, embedding_dim=200, batch_size=128):
    d = os.path.dirname(__file__)
    SENTENCE = Field(sequential=True, lower=True, include_lengths=False, batch_first=True, init_token='<sos>', eos_token='<eos>')
    train_ds = _load_dataset(SENTENCE, path=os.path.join(d, 'training.txt'))
    SENTENCE.build_vocab(train_ds, vectors="glove.6B.{}d".format(embedding_dim), specials=['<sos>'])

    valid_ds = _load_dataset(SENTENCE, path=os.path.join(d, 'validation.txt'))
    return (Iterator(valid_ds, batch_size=batch_size, train=False, device=device), SENTENCE.vocab)
     
def test_iterator(device, embedding_dim=200, batch_size=128):
    d = os.path.dirname(__file__)
    SENTENCE = Field(sequential=True, lower=True, include_lengths=False, batch_first=True, init_token='<sos>', eos_token='<eos>')
    train_ds = _load_dataset(SENTENCE, path=os.path.join(d, 'training.txt'))
    SENTENCE.build_vocab(train_ds, vectors="glove.6B.{}d".format(embedding_dim), specials=['<sos>'])

    test_ds = _load_dataset(SENTENCE, path=os.path.join(d, 'testing.txt'))
    return (Iterator(test_ds, batch_size=batch_size, train=False, device=device), SENTENCE.vocab)
    
def _load_dataset(field, path):
    fields = [("sentence", field)]

    with open(path, 'r', encoding='utf8') as f:
        examples = [Example.fromlist([line], fields) for line in f.readlines()]
            
        d = Dataset(examples, fields=fields)
        d.sort_key=lambda s: s.sentence
        
        return d