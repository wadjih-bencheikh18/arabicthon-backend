from helper.aroud_helper import get_wazn, get_aroud
from tachkil import get_tachkil


def get_full_aroud(line, selected_meter=''):
    """
    The ultimate function
    Input:
        line (str): the verse to process
    Output:
        meter: the meter of the verse
        aroud: the aroudi form of the verse
        tafil: the taf3ilat of the meter
        tachkil: the verse with predicted techkil
        harakat: the harakat ( 'O' and '|' ) of the verse
        ratio: the similarity between the verse harakat and the meter harakat

    Example:
        Input:
            line (str): كَم لَيلَةٍ عانَقتُ فيها غادَةً
        Output:
            meter (str): kaamil
            aroud: ['كَمْلَيْلَتِنْ', 'عَانَقْتُفِي', 'هَاغَادَتَنْ']
            tafil: ['مُتْفَاعِلُنْ', 'مُتْفَاعِلُنْ', 'مُتْفَاعِلُنْ']
            tachkil: كَمْ لَيْلَةٍ عَانَقْتُ فِيهَا غَادَةً
            harakat: ['|O|O||O', '|O|O||O', '|O|O||O']
            ratio: 0.6767676767
    """

    out = get_tachkil(line)
    tachkil, a = out['predicted'], out['aroudi']

    out2 = get_wazn(a, selected_meter=selected_meter)
    meter, wazn, ratio = out2['meter'], out2['wazn'], out2['ratio']

    out3 = get_aroud(a, wazn)
    aroud, tafil, harakat = out3['aroud'], out3['tafil'], out3['harakt']

    return {
        'meter': meter,
        'aroud': aroud,
        'tafil': tafil,
        'tachkil': tachkil,
        'harakat': harakat,
        'ratio': ratio
    }


poem = """قِف بِالمَنازِلِ إِن شَجَتكَ رُبوعُها
فَلَعَلَّ عَينَكَ تَستَهِلُّ دُموعُها
وَاِسأَل عَنِ الأَظعانِ أَينَ سَرَت بِها
آباؤُها وَمَتى يَكونُ رُجوعُها
دارٌ لِعَبلَةَ شَطَّ عَنكَ مَزارُها
وَنَأَت فَفارَقَ مُقلَتَيكَ هُجوعُها
فَسَقَتكِ يا أَرضَ الشَرَبَّةِ مُزنَةٌ
مُنهَلَّةٌ يَروي ثَراكِ هُموعُها
وَكَسا الرَبيعُ رُباكِ مِن أَزهارِهِ
حُلَلاً إِذا ما الأَرضُ فاحَ رَبيعُها
كَم لَيلَةٍ عانَقتُ فيها غادَةً
يَحيا بِها عِندَ المَنامِ ضَجيعُها
شَمسٌ إِذا طَلَعَت سَجَدتُ جَلالَةً
لِجَمالِها وَجَلا الظَلامَ طُلوعُها
يا عَبلَ لا تَخشَي عَلَيَّ مِنَ العِدا
يَوماً إِذا اِجتَمَعَت عَلَيَّ جُموعُها""".strip().split('\n')


for line in poem:
    out = get_full_aroud(line)
    meter = out['meter']
    aroud = out['aroud']
    tafil = out['tafil']
    tachkil = out['tachkil']
    harakat = out['harakat']

    print(line)
    print(meter)
    print(aroud)
    print(tafil)
    print(tachkil)
    print(harakat)

    print()
