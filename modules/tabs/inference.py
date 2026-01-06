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

        # ==== 上传 source_audio 文件 Column（内部使用，不返回）====
        with gr.Column():
            source_audio_file = gr.File(
                label="Upload Source Audio (override)",
                file_types=[".wav", ".mp3", ".flac"],
            )

    # 返回 **10 个值**，保持外部解包兼容
    return (
        source_audio_text,
        out_dir,
        transpose,
        embedding_model,
        embedding_output_layer,
        pitch_extraction_algo,
        auto_load_index,
        faiss_index_file,
        retrieval_feature_ratio,
        fo_curve_file,
    ), source_audio_file  # 返回控件对象用于内部 infer 使用


class Inference(Tab):
    def title(self):
        return "Inference"

    def sort(self):
        return 1

    def ui(self, outlet):
        def infer(
            sid,
            source_audio_text,
            out_dir,
            embedder_model,
            embedding_output_layer,
            f0_up_key,
            f0_file,
            f0_method,
            auto_load_index,
            faiss_index_file,
            index_rate,
            source_audio_file=None,
        ):
            model = models.vc_model
            try:
                yield "Infering...", None

                if out_dir == "":
                    out_dir = models.AUDIO_OUT_DIR

                # 上传文件存在则覆盖文本框路径
                if source_audio_file and source_audio_file != "":
                    if isinstance(source_audio_file, dict) and "name" in source_audio_file:
                        input_audio = source_audio_file["name"]
                    else:
                        input_audio = source_audio_file
                else:
                    input_audio = source_audio_text

                if not input_audio:
                    raise ValueError("No source audio provided")

                # 批量 / 文件夹 / 单文件逻辑
                if "*" in input_audio:
                    assert out_dir is not None, "Out folder is required for batch processing"
                    files = glob.glob(input_audio, recursive=True)
                elif os.path.isdir(input_audio):
                    assert out_dir is not None, "Out folder is required for batch processing"
                    files = glob.glob(
                        os.path.join(input_audio, "**", "*.wav"), recursive=True
                    )
                else:
                    files = [input_audio]

                audio = None
                for file in files:
                    audio = model.single(
                        sid,
                        file,
                        embedder_model,
                        embedding_output_layer,
                        f0_up_key,
                        f0_file,
                        f0_method,
                        auto_load_index,
                        faiss_index_file,
                        index_rate,
                        output_dir=out_dir,
                    )

                yield "Success", (model.tgt_sr, audio) if len(files) == 1 else None

            except Exception:
                yield "Error:\n" + traceback.format_exc(), None

        with gr.Group():
            with gr.Box():
                with gr.Column():
                    _, speaker_id = ui.create_model_list_ui()

                    (  # 注意这里解包仍是 10 个值
                        source_audio_text,
                        out_dir,
                        transpose,
                        embedder_model,
                        embedding_output_layer,
                        pitch_extraction_algo,
                        auto_load_index,
                        faiss_index_file,
                        retrieval_feature_ratio,
                        f0_curve_file,
                    ), source_audio_file = inference_options_ui()

                    with gr.Row(equal_height=False):
                        with gr.Column():
                            status = gr.Textbox(value="", label="Status")
                            output = gr.Audio(label="Output", interactive=False)

                    with gr.Row():
                        infer_button = gr.Button("Infer", variant="primary")

        # 点击按钮时传入组件对象
        infer_button.click(
            infer,
            inputs=[
                speaker_id,
                source_audio_text,
                out_dir,
                embedder_model,
                embedding_output_layer,
                transpose,
                f0_curve_file,
                pitch_extraction_algo,
                auto_load_index,
                faiss_index_file,
                retrieval_feature_ratio,
                source_audio_file,  # 内部覆盖文本框
            ],
            outputs=[status, output],
            queue=True,
        )

        return outlet()
