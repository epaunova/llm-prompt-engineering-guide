import gradio as gr
from evaluate import load

toxicity = load("toxicity")

def analyze_toxicity(text):
    score = toxicity.compute(predictions=[text])["toxicity"][0]
    return {"toxicity": round(score, 2)}

gr.Interface(
    fn=analyze_toxicity,
    inputs=gr.Textbox(label="Input Text"),
    outputs=gr.JSON(label="Toxicity Score"),
    title="🔍 Toxicity Analyzer for LLM Outputs"
).launch()
