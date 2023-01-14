from helper.helper_functions import get_sentence_tachkil, rebuild_sentence
from consts.awzan import AWZAN
from difflib import SequenceMatcher



def get_harakat(s, space=''):
    """
    Get the harakat of the verse
    'O' means soukoun
    '|' means haraka
    """

    res = ''

    letters, tachkil, _ = get_sentence_tachkil(s)

    for i, j in zip(letters, tachkil):
        if j == -1:
            if i == ' ':
                res += space
            else:
                res += 'O'
        elif j == 12:
            res += 'O'
        else:
            res += '|'

    return res


def get_aroud(text, wazn):
    """
    Get the kitaba aroudiya of a verse.
    Input:  
        text (str): either first part ( Sadr ) or second part ( Ajz ) of the verse
        wazn (str): the wazn of the meter ( Bahr )
    Output:
        aroud list(str): The kitaba aroudiya of the text
        tafil list(str): The taf3ilat of the meter's wazn
        harakt (str): the harakat ( 'O' and '|' ) of the text

    example:
        input:
            text: قِف بِالمَنازِلِ إِن شَجَتكَ رُبوعُها
            wazn: مُتْفَاعِلُنْ مُتَفَاعِلُنْ مُتَفَاعِلُنْ
        Output:
            aroud: ['قِفْبِلمَنَا', 'زِلِإِنْشَجَتْ', 'كَرُبُوعُهَا']
            tafil: ['مُتْفَاعِلُنْ', 'مُتَفَاعِلُنْ', 'مُتَفَاعِلُنْ']
            harakt: ['|O|O||O', '|||O||O', '|||O||O']

    """
    orig_h = list(get_harakat(text))
    wazn_h = get_harakat(wazn, space=' ')
    text = text.replace(' ', '')

    S, T, _ = get_sentence_tachkil(text)

    for i in range(len(wazn_h)):
        if wazn_h[i] == ' ':
            S.insert(i, ' ')
            T.insert(i, 14)
            orig_h.insert(i, ' ')

    aroud = rebuild_sentence(S, T)

    return {
        'aroud':aroud.split(' '),
        'tafil':wazn.split(' '),
        # 'harakt':wazn_h.split(' ')
        'harakt': ''.join(orig_h).split(' ')
    }



def get_wazn(text, selected_meter=''):
    """
    Find the most similiar meter and wazn for a given text
    Input:
        text (str): either first part ( Sadr ) or second part ( Ajz ) of the verse
        selected_meter (str): if equal to '', then we look for all awzan from all the meters ( Bohor )
            if not equal to '', we only look for the awzan that belong to the selected_meter 
    Output:
        meter (str): the meter of the text
        wazn (str): the taf3ilat of the wazn
        ratio (float): the similiarity between the harakat of the text and the harakat of the best meter's wazn

    Example:
        Input:
            text: قِف بِالمَنازِلِ إِن شَجَتكَ رُبوعُها
            selected_meter: ''
        Output:
            meter: kaamil
            wazn: مُتْفَاعِلُنْ مُتَفَاعِلُنْ مُتَفَاعِلُنْ
            ratio: 0.76767676767676

    """

    m = get_harakat(text)
    ratio_max = 0
    best = None

    for bahr in AWZAN:
        if (selected_meter == '') or (bahr == selected_meter):
            for mode in AWZAN[bahr]:
                for wazn in mode:
                    h = get_harakat(wazn)
                    ratio = SequenceMatcher(None, m, h).ratio()

                    if ratio > ratio_max:
                        ratio_max = ratio
                        best = (bahr, wazn)
    
    return {
                'meter':best[0],
                'wazn':best[1],
                'ratio':ratio,
            }
