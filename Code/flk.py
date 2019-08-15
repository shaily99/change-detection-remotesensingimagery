from flask import Flask, url_for, request, make_response
from flask import render_template
from model import InputForm
import sentinel as sn
import cv2
import base64
import requests
from datetime import datetime

app = Flask(__name__)
posts = [
    {
        'id': '1',
        'from': '31-01-18',
        'to': '31-04-18'
    },
    {
        'id': '2',
        'from': '31-05-18',
        'to': '31-09-18'
    }
]


# @app.route('/', methods=['GET', 'POST'])
# @app.route('/my', methods=['GET', 'POST'])
# def my():
#     form = InputForm(request.form)
#
#     return render_template('my.html', form=form)


@app.route('/', methods=['GET', 'POST'])
@app.route('/my', methods=['GET', 'POST'])
def my_post():
    form = InputForm(request.form)

    if request.method == 'POST':

        identity = form.ID.data
        sp1s = form.Span1_start.data
        sp1e = form.Span1_end.data
        sp2s = form.Span1_start.data
        sp2e = form.Span2_end.data

        img, rem, lat, lon = sn.get_image(id=identity, sp1s=sp1s, sp1e=sp1e, sp2s=sp2s, sp2e=sp2e)
    else:
        return render_template('my.html', form=form)

    dt1 = str(sp1e)
    dt2 = str(sp2e)
    query = {'plot_id': identity, 'lat': lat, 'lon': lon, 'remarks': rem,
             'source_image_date': dt1, 'target_image_date': dt2}

    req = requests.get(
        'https://egujforestgis.gujarat.gov.in/forest_mobile_web_test/change_detect', params=query, verify=False)

    url = req.url
    print(url)
    retval, buffer = cv2.imencode('.png', img)

    # return render_template('my.html', form=form, img=img)
    png_as_text = base64.b64encode(buffer)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/png'
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0')
