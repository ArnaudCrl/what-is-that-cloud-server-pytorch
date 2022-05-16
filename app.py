# -*-coding:utf-8 -*

import base64
from flask import Flask, request
from pathlib import Path
from pytorch_util import *
import dowload_model

app = Flask(__name__)


@app.route('/ping', methods=['POST'])
def wakeup():
    return "Hello World!"


@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        img_bytes = request.files['file'].read()
        # imgdata = base64.b64decode(img_bytes)
        tensor = transform_image(img_bytes)

        alexnet_prediction = get_alexnet_prediction(tensor)
        mobilenet_prediction = get_mobilenet_prediction(tensor)
        squeezenet_prediction = get_squeezenet_prediction(tensor)

        alexnet_pred_percent = torch.nn.functional.softmax(alexnet_prediction, dim=1).tolist()[0]
        mobilenet_pred_percent = torch.nn.functional.softmax(mobilenet_prediction, dim=1).tolist()[0]
        squeezenet_pred_percent = torch.nn.functional.softmax(squeezenet_prediction, dim=1).tolist()[0]

        average_percent = []
        for k in range(10):
            average = (alexnet_pred_percent[k] + mobilenet_pred_percent[k] + squeezenet_pred_percent[k]) / 3
            average_percent.append(average)

        return format_response(average_percent)


def format_response(probs):
    classes = ['altocumulus', 'altostratus', 'cirrocumulus', 'cirrostratus', 'cirrus', 'cumulonimbus', 'cumulus',
               'nimbostratus', 'stratocumulus', 'stratus']

    predictions = sorted(zip(classes, map(float, probs)), key=lambda p: p[1], reverse=True)

    result = '{{"{}":{}, '.format(predictions[0][0], predictions[0][1])
    for k in range(1, len(classes) - 1):
        result += '"{}":{}, '.format(predictions[k][0], predictions[k][1])
    result += '"{}":{}}}'.format(predictions[len(classes) - 1][0], predictions[len(classes) - 1][1])
    print(result)
    return result


if __name__ == '__main__':
    app.run()
