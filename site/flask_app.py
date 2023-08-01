import web_model
from flask import Flask, render_template, request

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/', methods=('GET','POST'))
def index():

    if request.method == 'POST':
        text = request.form['query']
        proba = web_model.ml_infer(text)
        rating, sentiment = web_model.estimate_rating(proba)
        web_model.compute_message_feature_importance(text, rating)
        if proba < 0.4:
            proba = 1-proba
        return render_template('base.html',
            query=text, proba=round(proba*100, 2),
            rating=rating, sentiment=sentiment, highlight=web_model.highlight)

    else:
        html = render_template('base.html', query=" ", proba=None, image=None)
        return html

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
