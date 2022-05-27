import re

TASHKIIL = [u"ِ", u"ُ", u"َ", u"ْ"]
TANWIIN = [u"ٍ", u"ٌ", u"ً"]
SHADDA = u"ّ"
# unknown diacretic (haraka)
UHARAKA = u"\u0653"

# Add shdda and madda
# Madda, in our case, is used to indicate an unknown haraka
DIAC = UHARAKA.join(SHADDA).join(TASHKIIL).join(TANWIIN)

# sun letters in arabic
SUN = u"[تثدذرزسشصضطظلن]"

# Sticky prepositions (bi-, li-)kasra? or (ka-, fa-, wa-)fatha?
# kasra and fatha can be madda in case there is no tashkiil
SPREP = u"[\u0644\u0628][\u0650%s]?|[\u0643\u0641\u0648][\u064E%s]?" % (UHARAKA, UHARAKA)

# alif in the middle of sentence
# DORJ = spaces or (bi-, li-)kasra? or (ka-, fa-, wa-)fatha?
DORJ = u"[^\\s]\\s+|%s" % SPREP

# ahruf al3illa: alif, alif maqsura, waw, yaa
ILLA = u"[اىوي]"

TATWEEL = {
    u"\u064E": u"\u064E\u0627",
    u"\u064F": u"\u064F\u0648",
    u"\u0650": u"\u0650\u064A",
}

CHANGE_LST = {
    u"هذا": u"هَاذَا",
    u"هذه": u"هَاذِه",
    u"هذان": u"هَاذَان",
    u"هذين": u"هَاذَين",
    u"ذلك": u"ذَالِك",
    u"ذلكما": u"ذَالِكُمَا",
    u"ذلكم": u"ذَالِكُم",
    u"الله": u"اللَّاه",
    u"اللهم": u"اللَّاهُمّ",
    u"إله": u"إِلَاه",
    u"الإله": u"الإِلَاه",
    u"إلهي": u"إِلَاهي",
    u"إلهنا": u"إِلَاهنا",
    u"إلهكم": u"إِلَاهكم",
    u"إلههم": u"إِلَاههم",
    u"إلههن": u"إِلَاههن",
    u"رحمن": u"رَحمَان",
    u"طاوس": u"طَاوُوس",
    u"داود": u"دَاوُود",
    u"لكن": u"لَاكِن",
    u"لكنني": u"لَاكِنَّنِي",
    u"لكنك": u"لَاكِنَّك",
    u"لكنه": u"لَاكِنَّه",
    u"لكنها": u"لَاكِنَّهَا",
    u"لكنهما": u"لَاكِنَّهُمَا",
    u"لكنهم": u"لَاكِنَّهُم",
    u"لكنهن": u"لَاكِنَّهُن",
    u"أولئك": u"أُلَائِك",
    u"أولئكم": u"أُلَائِكُم",
}

def modify(word):
    m = re.match(u"((?:%s)?)(.*)([%s]*)" % (SPREP, DIAC), word)
    begining = m.group(1)
    nodiac = re.sub(u"[%s]" % DIAC , "", m.group(2))
    ending = m.group(3)
    res = CHANGE_LST.get(nodiac, "")
    if res:
        return begining + res + ending

    return word

def normalize(text):
    res = text #result

    # Filtering
    # ===========
    #delete tatweel
    res = re.sub(u"\u0640", u"", res)

    #delete any non wanted char
    res = re.sub(u"[^\u0621-\u0652\\s]", u"", res)

    # Tashkiil
    # ===========

    # allati التِي
    res = res = re.sub(u"(^|\\s)\u0627\u0644\u0651?\u064E?\u062A\u0650\u064A(\\s|$)", u"\\1\u0627\u0644\u0651\u064E\u062A\u0650\u064A\\2", res)

    # if fatha damma or kasra is before shadda: switch
    res = res = re.sub(u"([\u064B-\u0650])\u0651", u"\u0651\\1", res)

    # add Fatha to first al-
    res = re.sub(u"^\\s*\u0627\u0644", u"\u0627\u064E\u0644", res)

    # Falty fathatan on alif fix
    res = re.sub(u"([^\\s])\u064E?([\u0627\u0649])\u064B", u"\\1\u064B\\2", res)

    # if alif is preceeding waw: add sukuun to alif
    res = re.sub(u"\u0627\u0648", u"\u0627\u0652\u0648", res)

    # repeat 2 times when there are two consecutive alif, etc.
    for i in range(2):
        # add Fatha to any non diacretized char preceeding alif X 2
        res = re.sub(u"([^\u064B-\u0650\\s])([\u0627\u0649])([^\u064B-\u0652]|$)", u"\\1\u064E\\2\\3", res)

        #add Damma to any non diacretized char preceeding waw
        res = re.sub(u"([^\u064B-\u0652\\s])\u0648([^\u064B-\u0652]|$)", u"\\1\u064F\u0648\\2", res)

        #add Kasra to any non diacretized char preceeding yaa
        res = re.sub(u"([^\u064B-\u0652\\s])\u064A([^\u064B-\u0652]|$)", u"\\1\u0650\u064A\\2", res)

    # add Shadda to shamsi characters after al-
    res = re.sub(u"(^|\\s)\u0627(\u064E?)\u0644(%s)([^\u0651])" % SUN, u"\\1\u0627\\2\u0644\\3\u0651\\4", res)

    # add madda to other characters after al-
    # res = re.sub(u"((?:^|\\s)\u0627\u0644[^\u0651])([^\u064E-\u0651])", u"\\1%s\\2" % UHARAKA, res)

    # add kasra to li
    res = re.sub(u"(^|\\s)\u0644([^\u064E-\u0652])", u"\\1\u0644\u0650\\2", res)

    # add kasra to bi
    res = re.sub(u"(^|\\s)\u0628([^\u064E-\u0652])", u"\\1\u0628\u0650\\2", res)

    # add fatha to fa
    res = re.sub(u"(^|\\s)\u0641([^\u064E-\u0652])", u"\\1\u0641\u064E\\2", res)

    # add fatha to wa
    res = re.sub(u"(^|\\s)\u0648([^\u064E-\u0652])", u"\\1\u0648\u064E\\2", res)

    # hamza under alif with no kasra
    res = re.sub(u"\u0625([^\u0650])", u"\u0625\u0650\\1", res)
    res = res = re.sub(u"\u0652([^\\s])([^\u064B-\u0650\\s])", u"\u0652\\1%s\\2" % UHARAKA, res)

    return res


def _prosody_del(text):
    res = text
    # Replace al- with sun character (it can be preceded by prepositions bi- li-)
    # والصِّدق، والشَّمس ---> وصصِدق، وَششَمس
    res = re.sub(u"(%s)\u0627\u0644(%s)" % (DORJ, SUN) , u"\\1\\2", res)
    res = re.sub(u"\u0627\u064E\u0644(%s)" % SUN , u"\u0627\u064E\\1", res)

    # Replace al- with l otherwise
    # # والكتاب، فالعلم ---> وَلكتاب، فَلعِلم
    res = re.sub(u"(%s)\u0627(\u0644[^\u064E-\u0650])" % DORJ, u"\\1\\2", res)


    # delete first alif of a word in the middle of sentence
    # فاستمعَ، وافهم، واستماعٌ، وابنٌ، واثنان ---> فَستَمَعَ، وَفهَم، وَستِماعُن، وَبنُن، وَثنانِ
    res = re.sub(u"(%s)\u0627([^\\s][^\u064B-\u0651\u0653])" % DORJ , u"\\1\\2", res)

    # delete ending alif, waw and yaa preceeding a sakin
    # أتى المظلوم إلى القاضي فأنصفه قاضي العدل ---> أتَ لمظلوم إلَ لقاضي فأنصفه قاضِ لعدل.
    res = re.sub(ILLA + u"\\s+(.[^\u064B-\u0651\u0653])", u" \\1", res)

    # delete alif of plural masculin conjugation
    # رجعوا ---> رجعو
    res = re.sub(u"[\u064F]?\u0648\u0627(\\s+|$)", u"\u064F\u0648\u0652\\1", res)

    return res

def _prosody_add(text):
    res = text

    #replace tanwiin taa marbuta by taa maftuuha
    res = re.sub(u"\u0629([\u064B-\u064D])", u"\u062A\\1", res)

    # delete alif from: fathatan + alif
    res = re.sub(u"\u064B(\u0627|\u0649)", u"\u064B", res)

    # Replace fathatan with fatha + nuun + sukuun
    res = re.sub(u"\u064B", u"\u064E\u0646\u0652", res)

    # Replace dammatun with damma + nuun + sukuun
    res = re.sub(u"\u064C", u"\u064F\u0646\u0652", res)

    # Replace kasratin with kasra + nuun + sukuun
    res = re.sub(u"\u064D", u"\u0650\u0646\u0652", res)

    # letter + Shadda ---> letter + sukuun + letter
    res = re.sub(u"(.)\u0651", u"\\1\u0652\\1", res)

    # hamza mamduuda --> alif-hamza + fatha + alif + sukuun
    res = re.sub(u"\u0622", u"\u0623\u064E\u0627\u0652", res)

    res = re.sub(u"([\u064E-\u0650])$", lambda m: TATWEEL[m.group(1)], res)

    return res

def _prosody_change(text):
    res = text
    res = re.sub(u"([^\s]+)", lambda m: modify(m.group(1)), res)
    return res



def prosody_form(text):
    res = text
    res = _prosody_change(res)
    res = _prosody_del(res)
    res = _prosody_add(res)
    return res