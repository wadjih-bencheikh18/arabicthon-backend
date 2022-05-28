import pandas as pd

import torch
import torchvision.transforms as T

from collections import Counter

from models.caption.model import EncoderDecoder

from PIL import Image
import requests
import matplotlib.pyplot as plt
import translators as ts
import spacy

from generation import generate_sentence

print('FINISH IMPORTS')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
spacy_eng = spacy.load("en_core_web_sm")



class Vocabulary:
    def __init__(self,freq_threshold):
        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}
        self.stoi = {v:k for k,v in self.itos.items()}
        self.freq_threshold = freq_threshold
        
    def __len__(self): 
        return len(self.itos)
    
    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]
    
    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                
                #add the word to the vocab if it reaches minum frequecy threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    

freq_threshold = 5
captions = pd.read_csv('models/caption/captions.txt')['caption'].to_list()
vocab = Vocabulary(freq_threshold)
vocab.build_vocab(captions)


#init caption_model
caption_model = EncoderDecoder(
    embed_size=300,
    vocab_size = len(vocab),
    attention_dim=256,
    encoder_dim=2048,
    decoder_dim=512,
    device=device
).to(device)


caption_model.load_state_dict(torch.load('models/caption/attention_model_state.pt', map_location=device)['state_dict'])


transform = T.Compose([
    T.Resize(226),                     
    T.ToTensor(),                               
    T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])


def generate_caption(url):
    """
    Giver a url of an image, we generate its arabic caption
    Input:
        url (str)L the url of the image
    Output:
        ar_caption (str): the arabic caption
    """

    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    img = transform(img).unsqueeze(0)

    caption_model.eval()
    with torch.no_grad():
        features = caption_model.encoder(img.to(device))
        caps, _ = caption_model.decoder.generate_caption(features,vocab=vocab)
        caption = ' '.join(caps[:-2])
        ar_caption = ts.google(caption, from_language='en', to_language='ar')

    return ar_caption


def generate_caption_sentence(url, max_lines):
    arabe_caption = generate_caption(url)
    res = generate_sentence(meter='الكامل', rhyme='ر',
                        max_lines=max_lines, start_with=arabe_caption)
    return res

    
arabe_caption = generate_caption('https://www.preventivevet.com/hubfs/Three%20dogs%20playing%20in%20the%20yard%20600%20canva.jpg')
res = generate_sentence(meter='الكامل', rhyme='ر', start_with=arabe_caption)
print(res)
