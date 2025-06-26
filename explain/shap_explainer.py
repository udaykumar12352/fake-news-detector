import shap
from transformers import pipeline

pipe = pipeline("text-classification", model="./model/best_model", tokenizer="./model/best_model", return_all_scores=True)

explainer = shap.Explainer(lambda x: [p[0]['score'], p[1]['score']] if (p := pipe(x)) else [0,0])

def shap_explain(text):
    shap_values = explainer([text])
    shap.plots.text(shap_values[0])
