vocab = [' ','ء', 'آ', 'أ', 'ؤ', 'إ', 'ئ', 'ا', 'ب', 'ة', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ى', 'ي']

vocab2token = { vocab[i]:i+1    for i in range(len(vocab)) }
token2vocab = { i+1:vocab[i]    for i in range(len(vocab)) }

special = ['َ', 'ُ', 'ِ', 'ً', 'ٌ', 'ٍ', 'َّ', 'ُّ', 'ِّ','ًّ', 'ٌّ', 'ٍّ', 'ْ','ّ']
sps_dict = {}
for i, c in enumerate(special):
    sps_dict[c] = i


ryhmes = ['ء', 'ب', 'ت', 'ح', 'د', 'ر', 'ع', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'س', 'ض', 'ي', 'ز', 'ش', 'ا', 'خ', 'و', 'ج', 'ص', 'ط', 'ث', 'ذ', 'ظ', 'غ', 'ى', 'هـ']
generation_meters = ['الخفيف', 'الكامل', 'الطويل', 'البسيط', 'الوافر', 'السريع', 'الرمل', 'المتقارب', 'الرجز']


classification_meters = ['الطويل','المنسرح','المتقارب','الخفيف','الكامل','السريع','الوافر','البسيط','الرجز','الرمل','المجتث','المديد','الهزج','المتدارك']
meter2id = {classification_meters[i]:i for i in range(len(classification_meters))}
id2meter = {i:classification_meters[i] for i in range(len(classification_meters))}


# Tachkil -----------------------------------------------------------------------------------------------------------------------

tachkil_vocab = [' ', '!', '"', '(', ')', '*', ',', '-', '.', ':', '?', '_', '«', '»', '،', '؛', '؟', 'ء', 'آ', 'أ', 'ؤ', 'إ', 'ئ', 'ا', 'ب', 'ة', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ى', 'ي', 'ٍ', '–', '…']

tachkil_vocab2token = { tachkil_vocab[i]:i+1    for i in range(len(tachkil_vocab)) }
tachkil_token2vocab = { i+1:tachkil_vocab[i]    for i in range(len(tachkil_vocab)) }

def tachkil_sentence2tokens(s):
    return [tachkil_vocab2token[c] if c in tachkil_vocab else len(tachkil_vocab) for c in s]

def tachkil_tokens2sentence(t):
    return [tachkil_vocab2token[c] for c in t]
