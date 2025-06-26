from lime.lime_text import LimeTextExplainer
from app.predictor import predict

class_names = ['Fake', 'Real']
explainer = LimeTextExplainer(class_names=class_names)

def lime_explain(text):
    def predict_proba(texts):
        return [predict(t)[1] for t in texts]

    explanation = explainer.explain_instance(text, predict_proba, num_features=10)
    explanation.show_in_notebook()
