import sys
import os
sys.path.insert(0, '.')

from flask import Flask, redirect, url_for, request, render_template
from flask_cors import CORS
from tachkil import get_tachkil
from taksim_aroud import get_full_aroud
# from generation import generate_sentence



app = Flask(__name__)
CORS(app)

# @app.route('/success/<name>')
# def success(name):
#    return 'welcome %s' % name

@app.route('/test')
def home():
    return render_template('login.html')


@app.route('/tachkil', methods=['POST'])
def tachkil():
    data = request.get_json()
    line = data['params']['text']
    res=[]
    for l in line:
        res.append(get_tachkil(l)["predicted"])

    print('\n'.join(res))
    return '\n'.join(res)


@app.route('/ultimateAroud', methods=['POST'])
def ultimateAroud():
    data = request.get_json()
    line = data['params']['text']

    res = []
    for l in line:
        res.append(get_full_aroud(l))

    res = {i:res[i] for i in range(len(res))}

    return res


# @app.route('/poemGeneration', methods=['POST'])
# def poemGeneration():
#     data = request.get_json()
#     meter = data['params']['meter']
#     rhyme = data['params']['rhyme']
#     lines = data['params']['lines']
#     s = generate_sentence(meter='الكامل', rhyme='ر', max_length=100)
#     return s


if __name__ == '__main__':
    app.run(debug=True)
