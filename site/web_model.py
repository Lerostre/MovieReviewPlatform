import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import nltk

from nltk.tokenize import word_tokenize
from plotly.subplots import make_subplots

logreg = pickle.load(open(f"logreg_0.905", 'rb'))
nltk.download('punkt')

def ml_infer(text):
    text = " ".join(word_tokenize(text))
    return logreg.predict_proba([text])[:, 1].item()


def estimate_rating(proba):
    rating = int(np.round(proba * 10))
    if rating >= 7:
        sentiment = "positive"
    elif rating <= 4:
        sentiment = "negative"
    else:
        sentiment = "mixed"
    return max(1, rating), sentiment

def compute_message_feature_importance(text, rating):

    _, inds = logreg[0].transform([text]).nonzero()
    important_words = logreg[0].get_feature_names_out()[inds]
    colors = ["#d20d39", "#c21a48", "#b52755", "#a83463", "#984171",
              "#8a4d7d", "#7C5B8C", "#6D689A", "#5E76A8", "#5281B4"]

    coefficients = logreg[1].coef_[0]
    feature_importance = pd.DataFrame({'feature': logreg[0].get_feature_names_out(),
                                       'abs_importance': np.abs(coefficients),
                                       'rel_importance': coefficients})
    feature_importance = feature_importance.sort_values('abs_importance',
                                                        ascending=False).set_index("feature")
    local_importance = feature_importance.loc[important_words]["rel_importance"]
    sub_df = local_importance.sort_values()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Predicted rating",
                                        "Word importance in given text"),
                        column_widths=[0.1, 0.9], horizontal_spacing = 0.05)

    fig.add_trace(
        go.Bar(x=sub_df.index, y=sub_df.values, showlegend=False,
               name="importance",
               marker=dict(
                color=sub_df.values,
                colorbar=dict(
                    title="importance"
                ),
                colorscale=[[0, 'crimson'], [1, 'steelblue']]
            )),
        row=1, col=2
    )

    fig.add_trace(
        go.Bar(y=[rating], showlegend=False, name="rating",
               marker_color=colors[rating-1]),
        row=1, col=1
    )

    fig.update_layout(yaxis_range=[-0.25, 10.25],
                      yaxis=dict(
                        tickvals=list(range(0, 11))), title_x=0.5,
                      height=600, autosize=True, template="simple_white",
                      font_size=16, font_family="Karma", title_font_family="Karma")

    if len(set(text.split())) > 15:
        showgrid = True
    else:
        showgrid = False
    fig['layout']["xaxis1"].update(showticklabels=False, mirror=True, showline=True)
    fig['layout']["yaxis1"].update(tickvals=list(range(0, 11)), title="rating", mirror=True, showline=True)
    fig['layout']["xaxis2"].update(tickangle=-45, title="feature",
                                   mirror=True, showline=True, showgrid=showgrid)
    fig['layout']["yaxis2"].update(title="", showgrid=False, mirror=True, showline=True)
    fig.update_annotations(yshift=20)
    fig.layout.annotations[0].update(font_size=16)
    fig.layout.annotations[1].update(font_size=24)
    fig.write_html("mysite/templates/temp.html")


def highlight(string):
    return f'<mark_{string}>{string}</mark_{string}>'