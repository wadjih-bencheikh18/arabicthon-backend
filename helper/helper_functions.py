from consts.static import *

# Generation + Classification function ---------------------------------------------------------------------------------------------------

def sentence2tokens(s):
    """
    Transform a sentence into a list of tokens
    """
    return [vocab2token[c] if c in vocab else len(vocab) for c in s]

def tokens2sentence(t):
    """
    Transform a lsit of tokens into a sentence
    """
    return [token2vocab[c] for c in t]

def get_raw_sentence(sentence):
    """
    Get a raw sentence without tachkil
    """
    
    sentence = list(sentence)

    S = []

    i, j = 0, 1
    L = len(sentence)

    while i < L - 1:
        if sentence[i] == ' ':
            S.append(' ')
            i += 1
            j += 1

        else:
            S.append(sentence[i])
            if sentence[j] not in special:
                i += 1
                j += 1

            elif sentence[j] != 'ّ':
                i += 2
                j += 2

            else:  # sentence[j] = 'ّ'
                if j == L-1 or sentence[j+1] not in special:
                    i += 2
                    j += 2
                else:
                    i += 3
                    j += 3

    if i < L and not sentence[-1] in special:
        S.append(sentence[-1])

    return S


# Tachkil -----------------------------------------------------------------------------------------------------------------------

def get_sentence_tachkil(sentence):
    """
    Get the letters, tachkil of a sentence along with the tachkil ratio of the sentence
    """

    sentence = list(sentence)

    T = []
    S = []

    cnt = 0

    i, j = 0, 1
    L = len(sentence)

    while i < L - 1:
        if sentence[i] == ' ':
            S.append(' ')
            T.append(-1)
            i += 1
            j += 1

        else:
            S.append(sentence[i])
            if sentence[j] not in special:
                T.append(-1)
                i += 1
                j += 1

            elif sentence[j] != 'ّ':
                T.append(sps_dict[sentence[j]])
                cnt += 1
                i += 2
                j += 2

            else:  # sentence[j] = 'ّ'
                if j == L-1 or sentence[j+1] not in special:
                    T.append(sps_dict['ّ'])
                    cnt += 1
                    i += 2
                    j += 2
                else:
                    try:
                        T.append(sps_dict['ّ' + sentence[j+1]])
                    except:
                        T.append(sps_dict['ّ'])
                    cnt += 1
                    i += 3
                    j += 3

    if i < L:
        if sentence[-1] in special:
            T.append(sps_dict[sentence[-1]])
            cnt += 1
        else:
            S.append(sentence[-1])
            T.append(-1)

    return S, T, cnt / len(S)

def rebuild_sentence(S, T, real_T=None):
    """
    Rebuild the sentence from the letters and the tachkil.
    This function is used after we predict the tachkil of the letter.
    Real_T is a list that contains the original diacritics of the sentence. If available, we won't replace it with the predicted tachkila
    """

    res = ''

    if real_T is None:
        for c, t in zip(S, T):
            if t == 14 or t == -1:
                res += c
            else:
                res = res + c + special[t]


    else:
        for c, t, rt in zip(S, T, real_T):
            if rt != -1:
                res = res + c + special[rt]
            else:
                if t == 14 or t == -1:
                    res += c
                else:
                    res = res + c + special[t]

    return res

def tachkil_sentence2tokens(s):
    """
    Transform a sentence into a list of tokens for tachkil task
    """
    return [tachkil_vocab2token[c] if c in tachkil_vocab else len(tachkil_vocab) for c in s]

def tachkil_tokens2sentence(t):
    """
    Transform a lsit of tokens into a sentence for tachkil task
    """
    return [tachkil_vocab2token[c] for c in t]
