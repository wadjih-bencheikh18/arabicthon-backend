from generation import generate_sentence
from helper.helper_functions import get_raw_sentence



def get_last_word(right, left):
    """
    Generated the last word of the verse given the right part ( Sadr ) and the left part ( AJZ )
    return:
        res (str): the generated verse
    """
    
    right = get_raw_sentence(right)
    left = get_raw_sentence(left)

    right, left = ''.join(right), ''.join(left) 

    s = right + ' _ ' + left
    s = ' * ' + s

    res = generate_sentence(meter='الكامل', rhyme='ر', start_with=s, max_lines=1, max_length=50)
    
    return res


# right = 'عرج بمنعرج الكثيب الأعفر'
# left = 'بين الفرات وبين شط'
# print(get_last_word(right, left))