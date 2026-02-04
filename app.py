"""
GI-NET Ki-67 Grade Prediction Web Application.

Hugging Face Spaces deployment for predicting Ki-67 proliferation grade
(G1 vs G2+G3) from H&E histopathology images.
"""

import os

import gradio as gr

from predictor import GINETPredictor

# Configuration - weights will be downloaded from HF Hub automatically
MODEL_REPO = os.environ.get("MODEL_REPO", "")  # e.g., "username/gi-net-weights"
MODEL_DIR = os.environ.get("MODEL_DIR", "./models/ABMIL_binary")
DEVICE = os.environ.get("DEVICE", "cuda")

# Global predictor instance (lazy loaded)
predictor = None


def load_model() -> GINETPredictor:
    """Load the predictor model (lazy initialization)."""
    global predictor
    if predictor is None:
        predictor = GINETPredictor(
            model_dir=MODEL_DIR,
            model_repo=MODEL_REPO if MODEL_REPO else None,
            device=DEVICE,
        )
    return predictor


def predict_grade(image) -> tuple[str, str, str]:
    """
    Gradio prediction function.

    Args:
        image: Uploaded image (PIL Image from Gradio)

    Returns:
        Tuple of (prediction_text, probability_text, interpretation_text)
    """
    if image is None:
        return "Please upload an image.", "", ""

    try:
        pred = load_model().predict(image)

        # Format results for display
        result_text = f"## {pred['prediction']}\n\n**Confidence:** {pred['confidence']*100:.1f}%"

        prob_text = f"""
### Class Probabilities

| Grade | Probability |
|-------|-------------|
| G1 (Ki-67 <3%) | {pred['prob_g1']*100:.1f}% |
| G2+G3 (Ki-67 ≥3%) | {pred['prob_g2g3']*100:.1f}% |

*{pred['n_tiles']} tissue region(s) analyzed*
"""

        return result_text, prob_text, pred["interpretation"]

    except FileNotFoundError as e:
        return f"**Model Error:** {str(e)}", "", ""
    except Exception as e:
        return f"**Error:** {str(e)}", "", ""


# Create the Gradio interface
with gr.Blocks(
    title="GI-NET Ki-67 Grade Prediction",
    theme=gr.themes.Soft(),
    css="""
    .main-title {
        text-align: center;
        margin-bottom: 1rem;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    """,
) as demo:
    gr.Markdown(
        """
        # 🔬 GI-NET Ki-67 Grade Prediction from H&E

        Upload an H&E histopathology image to predict Ki-67 proliferation grade using AI.

        **Supported grades:**
        - **G1**: Ki-67 <3% (low proliferation)
        - **G2+G3**: Ki-67 ≥3% (intermediate/high proliferation)

        **Model Performance:** 94.9% accuracy | 0.90 Cohen's kappa | 97% G1 sensitivity | 93% G2+G3 sensitivity
        """,
        elem_classes=["main-title"],
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload H&E Image",
                sources=["upload", "clipboard"],
                height=400,
            )
            predict_btn = gr.Button("🔍 Predict Grade", variant="primary", size="lg")

        with gr.Column(scale=1):
            prediction_output = gr.Markdown(label="Prediction")
            probability_output = gr.Markdown(label="Probabilities")

    interpretation_output = gr.Markdown(label="Clinical Interpretation")

    # Connect button to prediction function
    predict_btn.click(
        fn=predict_grade,
        inputs=[image_input],
        outputs=[prediction_output, probability_output, interpretation_output],
    )

    # Also trigger prediction on image upload
    image_input.upload(
        fn=predict_grade,
        inputs=[image_input],
        outputs=[prediction_output, probability_output, interpretation_output],
    )

    gr.Markdown(
        """
        ---

        <div class="disclaimer">

        **⚠️ Disclaimer:** This tool is for **research and clinical decision support only**.
        Not intended for primary diagnosis. Final grading decisions should be made by a
        qualified pathologist. Consider Ki-67 IHC when clinically indicated.

        </div>

        ---

        **About:** This application uses an Attention-Based Multiple Instance Learning (ABMIL)
        model with H-optimus-0 feature extraction to predict Ki-67 grade from H&E images.
        The model was trained on GI-NET histopathology data using 5-fold cross-validation.
        """
    )

# For Hugging Face Spaces
if __name__ == "__main__":
    demo.launch()
