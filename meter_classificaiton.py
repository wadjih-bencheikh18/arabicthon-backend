import torch
import torch.nn as nn

from helper.helper_functions import get_raw_sentence, sentence2tokens
from consts.static import vocab, classification_meters, id2meter
from models.meter_classification.model import LstmModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VOCAB_SIZE = len(vocab) + 1
OUTPUT_SIZE = len(classification_meters)

meter_classificaiton_model = LstmModel(
    128,
    VOCAB_SIZE,
    OUTPUT_SIZE).to(device)


meter_classificaiton_model.load_state_dict(torch.load(
    'models/meter_classification/meter_classification_model.pt', map_location=device))


poem = """عاوَدَ عَينَي نَصبُهَا وَغُرورُها
أهمُّ عَناهَا أمْ قَذَاها يَعُورُهَا
أم الدار أَمسَت قَد تَعَفَّت كأَنَّهَ
زَبُورُ يَمَانٍ نَقَّشَتهُ سُطُورُهَا
ذَكَرَتُ بِها هِنداً وأترابَها الأُلَى
بِهَا يُكذَبُ الوَاشي ويُعصَى أميرُهَا
فَمَا مُعولٌ تَبكِي لِفَقدِ أَلِيفِهَا
إِذَا ذَكَرَتهُ لاَ يَكُفُّ زَفِيرُها
بِأَغْزرَ مِنّي عَبرةً إِذَ رَأيتُهَا
يحَثُ بِهَا قَبلَ الصَّبَاحِ بَعيرُهَا
أَلَم يَأتِ هِنداً كَيفَما صُنعُ قَومِه
بَني عَامِر إِذا جَاءَ يَسعَى نَذِيرُهَا
فَقَالُوا لَنَا إِنَّا نُحِبُّ لِقَاءَكُم
وَأَنَّا نُحَيِّيي أرضَكُم ونَزُورُها
فَقُلنَا إِذن لاَ نَنكُلُ الدَّهرَ عنكُمُ
بِصُمِّ القَنَا اللاَّئِي الدِّمَاءَ تُميرُها
فَلاَ غَروَ أَنَّ الخَيلَ تَنحَطُ في القَنَا
تَمَطَّرُ مِن تَحتِ العوالي ذُكُورها
تَأَوَّهُ مِمّا مَسَّها مِن كَرِيهَةٍ
وتُصغِىِ الخُدُودَ وَالرِّمَاحُ تَصُورُها
وأَرْبابُها صَرعى بِبُرقَةِ أَخرَبٍ
تُجرِّرُهُم ضِبعَانُها ونُسورُها
فَأبلِغ أَبَا الحَجَّاجِ عَنِّي رِسَالَةً
مُغلغَلةً لاَ يُفلِتَنكَ بُسُورُها
فَأَنتَ مَنَعتَ السَّلمَ يَومَ لَقِيتَنَا
بِكفِّيك تُسدِي غَيَّةً وتُنيرُهَا
فَذُوقُوا عَلَى مَا كَانَ مِن فَرط إحنَةٍ
حَلائِبَنَا إِذ غَابَ عَنَّا نَصِيرُها""".strip().split('\n')


def predict_meter(right, left, k=3, temp=2):
    """
    Predict the meter of a raw (without tachkil) sentence
    Input:
        right (str): the Sadr
        left (str): the Ajz
        k (int): the top k meters
        temp (float): the temperature fot the softmax
    Output:
        return list(tuple(float, str)): a list of the top k meters along with there probabilities
    """

    meter_classificaiton_model.eval()

    right = get_raw_sentence(right)
    left = get_raw_sentence(left)

    s = ''.join(right) + ' # ' + ''.join(left)

    x = sentence2tokens(s)
    x = torch.tensor(x, dtype=torch.int64)

    x = nn.ConstantPad1d((0, 100 - x.shape[0]), 0)(x).unsqueeze(0).to(device)

    pred = meter_classificaiton_model(x).cpu()[0]
    pred = pred / temp
    pred = torch.softmax(pred, dim=-1, )

    args = list(torch.argsort(pred, dim=-1, descending=True))[:k]
    args = [i.item() for i in args]

    probs = [pred[i].item() for i in args[:k]]

    return [(id2meter[args[i]], probs[i]) for i in range(k)]


# for i in range(len(poem) // 2):
#     right = poem[i*2].strip()
#     left = poem[i*2 + 1].strip()
#     print(predict_meter(right, left))
