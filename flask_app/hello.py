# fix path for imports
import sys
import os
sys.path.insert(0, '.')
# imports 
from last_word_prediction import get_last_word
from generation import generate_sentence
from caption_generation import generate_caption_sentence
from taksim_aroud import get_full_aroud
from meter_classificaiton import predict_meter
from tachkil import get_tachkil
from flask_cors import CORS
from flask import Flask, redirect, url_for, request, render_template
import base64
import io
from PIL import Image


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
    res = []
    result = []
    for l in line:
        res.append(get_tachkil(l)["predicted"])

    if len(res) % 2 == 0:
        for i in range(len(res) // 2):
            right = res[i*2].strip()
            left = res[i*2 + 1].strip()
            result.append(right+"*"+left)
    else:
        result = res

    return '\n'.join(result)


@app.route('/meter', methods=['POST'])
def meter():
    data = request.get_json()
    line = data['params']['text']
    res = []
    for i in range(len(line) // 2 if len(line) > 1 else 1):
        right = line[i*2].strip()
        left = line[i*2 + 1].strip() if i*2 + 1 < len(line) else ""
        print(right, left)
        res.append(predict_meter(right, left))

    return {i: res[i] for i in range(len(res))}


@app.route('/ultimateAroud', methods=['POST'])
def ultimateAroud():
    data = request.get_json()
    line = data['params']['text']
    res = []
    print(line)
    for l in line:
        res.append(get_full_aroud(l))

    res = {i: res[i] for i in range(len(res))}

    return res


@app.route('/poemGeneration', methods=['POST'])
def poemGeneration():
    data = request.get_json()
    meter = data['params']['meter']
    rhyme = data['params']['rhyme']
    lines = int(data['params']['lines'])
    sujet = data['params']['sujet']
    s = generate_sentence(meter, rhyme, lines,
                          start_with=sujet, max_length=lines*50)
    return s


@app.route('/caption', methods=['POST'])
def caption():
    data = request.get_json()
    file = data['params']['file']
    msg = base64.b64decode(file.split(",")[1])
    # print(msg)
    buf = io.BytesIO(msg)
    img = Image.open(buf).convert("RGB")
    lines = int(data['params']['lines'])
    rhyme = data['params']['rhyme']
    
    s = generate_caption_sentence(img, lines, rhyme)
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
    app.run(debug=False, port=4000)
