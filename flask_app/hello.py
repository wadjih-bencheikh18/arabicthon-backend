import base64
import sys
import os
sys.path.insert(0, '.')
import cv2
import numpy
from flask import Flask, redirect, url_for, request, render_template
from flask_cors import CORS
from tachkil import get_tachkil
from meter_classificaiton import predict_meter
from taksim_aroud import get_full_aroud
from caption_generation import generate_caption_sentence
from generation import generate_sentence
from last_word_prediction import get_last_word


app = Flask(__name__)
CORS(app)

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
    res=[]
    result = []
    for l in line:
        res.append(get_tachkil(l)["predicted"])

    for i in range(len(res) // 2):
        right = res[i*2].strip()
        left = res[i*2 + 1].strip()
        result.append(right+"*"+left)

    return '\n'.join(result)


@app.route('/meter', methods=['POST'])
def meter():
    data = request.get_json()
    line = data['params']['text']
    res = []  
    for i in range(len(line) // 2):
        right = line[i*2].strip()
        left = line[i*2 + 1].strip()
        res.append(predict_meter(right, left))

    return {i: res[i] for i in range(len(res))}


@app.route('/ultimateAroud', methods=['POST'])
def ultimateAroud():
    data = request.get_json()
    line = data['params']['text']
    res = []
    for l in line:
        res.append(get_full_aroud(l))

    res = {i:res[i] for i in range(len(res))}

    return res


@app.route('/poemGeneration', methods=['POST'])
def poemGeneration():
    data = request.get_json()
    meter = data['params']['meter']
    rhyme = data['params']['rhyme']
    lines = int(data['params']['lines'])
    sujet = data['params']['sujet']
    s = generate_sentence(meter, rhyme, lines, start_with=sujet, max_length=400)
    return s


@app.route('/caption', methods=['POST'])
def caption():
    data = request.get_json()
    image = data['params']['file']
    lines = int(data['params']['lines'])
    s = generate_caption_sentence(image, lines)
    return s


@app.route('/lastword', methods=['POST'])
def lastword():
    data = request.get_json()
    right = data['params']['right']
    left = data['params']['left']
    meter = data['params']['meter']
    rhyme = data['params']['rhyme']
    s = get_last_word(right, left, meter, rhyme)
    return s

if __name__ == '__main__':
    app.run(debug=True)
