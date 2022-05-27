import sys
import os
sys.path.insert(0, '.')

from flask import Flask, redirect, url_for, request, render_template
from tachkil import get_tachkil
from taksim_aroud import get_full_aroud
# from generation import generate_sentence



app = Flask(__name__)


# @app.route('/success/<name>')
# def success(name):
#    return 'welcome %s' % name

@app.route('/')
def home():
    return render_template('login.html')


@app.route('/tachkil', methods=['POST'])
def tachkil():
    data = request.get_json()
    line = data['params']['text']
    res = get_tachkil(line)
    return res


@app.route('/ultimateAroud', methods=['POST'])
def ultimateAroud():
    data = request.get_json()
    line = data['params']['text']
    res = get_full_aroud(line)
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
