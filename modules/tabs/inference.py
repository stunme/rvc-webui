import glob
import os
import traceback

import gradio as gr

from modules import models, ui
from modules.ui import Tab


def inference_options_ui(show_out_dir=True):
    with gr.Row(equal_height=False):
        # ==== 原 source_audio + out_dir Column ====
        with gr.Column():
            source_audio_text = gr.Textbox(
                label="Source Audio",
                placeholder="Path / wildcard / directory",
            )
            out_dir = gr.Textbox(
                label="Out folder",
                visible=show_out_dir,
                placeholder=models.AUDIO_OUT_DIR,
            )

        # ==== Transpose + pitch + embedder Column ====
        with gr.Column():
            transpose = gr.Slider(
                minimum=-20, maximum=20, value=0, step=1, label="Transpose"
            )
            pitch_extraction_algo = gr.Radio(
                choices=["dio", "harvest", "mangio-crepe", "crepe"],
                value="crepe",
                label="Pitch Extraction Algorithm",
            )
            embedding_model = gr.Radio(
                choices=["auto", *models.EMBEDDINGS_LIST.keys()],
                value="auto",
                label="Embedder Model",
            )
            embedding_output_layer = gr.Radio(
                choices=["auto", "9", "12"],
                value="auto",
                label="Embedder Output Layer",
            )

        # ==== Index Column ====
        with gr.Column():
            auto_load_index = gr.Checkbox(value=False, label="Auto Load Index")
            faiss_index_file = gr.Textbox(value="", label="Faiss Index File Path")
            retrieval_feature_ratio = gr.Slider(
                minimum=0,
                maximum=1,
                value=1,
                step=0.01,
                label="Retrieval Feature Ratio",
            )

        # ==== F0 Curve 文件 Column ====
        with gr.Column():
            fo_curve_file = gr.File(label="F0 Curve File")

        # ==== 单独 Column 上传 source_audio 文件 ====
        with gr.Column():
            source_audio_file = gr.File(
                label="Upload Source Audio (override)",
                file_types=[".wav", ".mp3", ".flac"],
            )

    # 返回组件对象，外部解包兼容
    return (
        source_audio_text,
        out_dir,
        transpose,
        embedding_model,
        embedding_output_layer,
        pitch_extraction_algo,
        auto_load_index,
        faiss_index_file,
        retrieval_feature_ratio
