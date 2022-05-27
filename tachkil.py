import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from consts.static import tachkil_vocab
from helper.helper_functions import get_sentence_tachkil, tachkil_sentence2tokens, rebuild_sentence
from models.tachkil.model import TachkilLstmModel
from helper.prosody_postprocessing import prosody_form, normalize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


special = ['َ', 'ُ', 'ِ', 'ً', 'ٌ', 'ٍ', 'َّ',
           'ُّ', 'ِّ', 'ًّ', 'ٌّ', 'ٍّ', 'ْ', 'ّ']
sps_dict = {}
for i, c in enumerate(special):
    sps_dict[c] = i



VOCAB_SIZE = len(tachkil_vocab)+1  # 0 is for the padding
OUTPUT_SIZE = len(special)+1  # +1 for the non tachkil

model_tachkil = TachkilLstmModel(
    256,
    VOCAB_SIZE,
    OUTPUT_SIZE).to(device)


model_tachkil.load_state_dict(torch.load(
    'models/tachkil/tachkil_model.pt', map_location=device))


poem = """كَأَنَّما البَصرَةُ مُذ جُزتَها
أَشلاءُ غِمدٍ سُلَّ مِنهُ الحُسام
بُدِّلَتِ الوَحشَة مِن أُنسِها
وَأُلبِسَت بَعدَ الضِياءِ الظَلام
وَكُلُّ أَرضٍ فَقَدَت شَمسَها
غَشّى الدُجى ساحاتِها وَالخِيام
وَالآنَ إِذ عُدتَ إِلَيها زَهَت
وَاِبتَسَمَت بِالبِشرِ أَيَّ اِبتِسام
فَلا خَلَت مِنكَ الرَعايا وَلا
زِلتَ لِمُرفَضِّ المَعالي نِظام""".strip().split('\n')



def get_tachkil(line):
    """
    Predict the tachkil on a given text
    Input:  
        line (str): the verse to tachkil
    Output:
        predicted (str): the new verse with tachkil
        aroudi (str): the aroudi form of the verse
    """

    S, T, _ = get_sentence_tachkil(line)

    x = tachkil_sentence2tokens(S)
    x = torch.tensor(x, dtype=torch.int64)
    L = len(x)
    x = nn.ConstantPad1d((0, 50 - x.shape[0]), 0)(x)
    x = x.unsqueeze(0).to(device)

    out = model_tachkil(x)
    out = torch.argmax(out, dim=-1)[0, :L].cpu()
    out = list(np.array(out))

    out = rebuild_sentence(S, out, T)

    norm_out = normalize(out)
    prosody_out = prosody_form(norm_out)

    return {
        'predicted': norm_out,
        'aroudi'   : prosody_out
    }



# for line in poem:
#     out = get_tachkil(line)
#     print('predicted', out['predicted'])
#     print('aroudi'   , out['aroudi'])
#     print()
