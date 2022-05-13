# -*-coding:utf-8 -*

import base64
import aioflask
from flask import Flask
from pathlib import Path
from pytorch_util import *
import dowload_model

app = Flask(__name__)


@app.route('/wakeup', methods=['POST'])
async def wakeup(request):
    return "Hello World!"


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    print("connecting")
    img_data = await request.form()
    img_bytes = await (img_data['file'])
    imgdata = base64.b64decode(str(img_bytes))
    tensor = transform_image(imgdata)
    prediction = get_prediction(tensor)
    pred_percent = torch.nn.functional.softmax(prediction, dim=1)
    return format_response(pred_percent.tolist()[0])


def format_response(probs):
    # classes = ['altocumulus', 'altostratus', 'cirrocumulus', 'cirrostratus', 'cirrus', 'cumulonimbus', 'cumulus',
    #            'nimbostratus', 'stratocumulus', 'stratus']
    classes = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012',
               '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024',
               '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
               '037', '038', '039', '040', '041', '042', '043', '044', '045', '046']

    predictions = sorted(zip(classes, map(float, probs)), key=lambda p: p[1], reverse=True)

    result = '{{"{}":{}, '.format(predictions[0][0], predictions[0][1])
    for k in range(1, len(classes) - 1):
        result += '"{}":{}, '.format(predictions[k][0], predictions[k][1])
    result += '"{}":{}}}'.format(predictions[len(classes) - 1][0], predictions[len(classes) - 1][1])

    return result


if __name__ == '__main__':
    app.run()
