import torch

from transformers import GPT2TokenizerFast, pipeline
from transformers import GPT2LMHeadModel
from transformers import AutoConfig

print('FINISH IMPORTS')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_NAME = 'aubmindlab/aragpt2-base'
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)

config = AutoConfig.from_pretrained(MODEL_NAME)
generation_model = GPT2LMHeadModel(config=config)

print('FINISH BUILDING THE MODEL')

generation_model.load_state_dict(torch.load(
    'models/generation/aragpt2.pt', map_location=device))

print('FINISH LOADING THE MODEL')

generation_pipeline = pipeline(
    "text-generation", model=generation_model, tokenizer=tokenizer)


def generate_sentence(meter, rhyme, max_lines=100, max_length=250, start_with='', num_beams=5):
    """
    Generate poem given a meter, rhyme and a start off text
    Input:
        meter (str): the Bahr
        rhyme (str): the rhyme or 7arf rawi
        max_lines (int): the max number of verses in the generated poems
        max_length (int): the max number of tokens in the generated poems
        start with (str): a text to start with
    Output (str):
        return the generated text
    """

    start_with = start_with.strip()
    x = f'<|endoftext|>{meter}<|endoftext|>{rhyme}<|endoftext|>{start_with}'
    out = generation_pipeline(x,
                              # pad_token_id=tokenizer.eos_token_id,
                              num_beams=num_beams,
                              max_length=max_length,
                              top_p=0.9,
                              repetition_penalty=3.0,
                              no_repeat_ngram_size=2, device=0)[0]

    out = out['generated_text'][46:]

    i = 0
    res = []
    for line in out.split(' * '):
        if '_' in line:
            r, l = line.split(' _ ')
            r, l = r.strip(), l.strip()
            res += [r + ' _ ' + l]
        else:
            res += [line]
        i += 1
        if i == max_lines:
            break
    
    return '\n'.join(res)






# s = generate_sentence(meter='الكامل', rhyme='ر', start_with='')
# print(s)
