#coding:utf-8
import random
import time
import argparse
import requests
import gradio as gr
from PIL import Image
from io import BytesIO

import torch
from transformers import TextStreamer

from bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from bunny.conversation import conv_templates, SeparatorStyle
from bunny.model.builder import load_pretrained_model
from bunny.util.utils import disable_torch_init
from bunny.util.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, \
    KeywordsStoppingCriteria


parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default='/mount/01models/Bunny-v1.0-3B/BAAI/Bunny-v1_0-3B')
parser.add_argument("--model-base", type=str, default=None)
parser.add_argument("--model-type", type=str, default='phi-2')
parser.add_argument("--image-file", type=str, required=False,default='/mount/01models/Bunny-v1.0-3B/000000006652.jpg')
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--conv-mode", type=str, default=None)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--load-8bit", action="store_true")
parser.add_argument("--load-4bit", action="store_true")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()
print('gradio version:',gr.__version__)
disable_torch_init()

model_name = get_model_name_from_path(args.model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                                       args.model_type, args.load_8bit,
                                                                       args.load_4bit, device=args.device)
conv_mode = "bunny"

if args.conv_mode is not None and conv_mode != args.conv_mode:
    print(
        '[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode,
                                                                                                          args.conv_mode,                                                                                                   args.conv_mode))
else:
    args.conv_mode = conv_mode

conv = conv_templates[args.conv_mode].copy()
roles = conv.roles

def reset_state():
    return None, [], []
def reset_user_input():
    return gr.update(value='')


def predict(input_text,image_path_upload, chatbot,max_new_tokens, top_p,temperature,history,selected):
    if selected == 'Upload':
        image_path = image_path_upload
    print('image_path',image_path)
    if image_path is None:
        return [(input_text, "图片不能为空。请重新上传图片。")], []

    image_tensor = process_images([image_path], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=model.dtype)

    print('***************** chatbot ***************')
    print('chatbot', chatbot)
    print('chatbot', dir(chatbot))
    print('history', len(history))
    print('*' * 100)
    if len(chatbot) == 0:
        # first message
        if len(conv.messages):
            conv.messages = []
        inp = DEFAULT_IMAGE_TOKEN + '\n' + input_text
        conv.append_message(conv.roles[0], inp)
        # image = None
    else:
        conv.append_message(conv.roles[0], input_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print('***************** prompt ***************')
    print(prompt)
    print('*' * 100)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
        model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    conv.messages[-1][-1] = outputs

    if args.debug:
        print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
    outputs = outputs.split('<|endoftext|>')[0]
    temp = [input_text,outputs] ## input,response
    history.append(temp)
    chatbot.append(temp)
    print('history',history)
    print('**************** chatbot',chatbot)
    yield chatbot, history

title_markdown = ("""
# 🐰 Bunny: A family of lightweight multimodal models
""")
with gr.Blocks(theme=gr.themes.Default()) as demo:

    selected_state = gr.State("Upload")
    gr.Markdown(title_markdown)

    with gr.Row():

        with gr.Column(scale=2.5):
            image_path_upload = gr.Image(type="pil", label="Image", value=None,height=310)
            max_new_tokens = gr.Slider(0, 1024, value=512, step=1.0, label="Max new tokens", interactive=True)
            top_p = gr.Slider(0, 1, value=0.9, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.5, step=0.01, label="Temperature", interactive=True)
        with gr.Column(scale=3.5):
            chatbot = gr.Chatbot(height=400)
            user_input = gr.Textbox(show_label=False, placeholder="Your Instruction here", lines=4,container=False)
            with gr.Row():
                submitBtn = gr.Button("提交", variant="primary")
                emptyBtn = gr.Button("清除")

    history = gr.State([])
    submitBtn.click(predict,
                    [user_input,image_path_upload, chatbot,max_new_tokens, top_p,temperature,history,selected_state], [chatbot, history],
                    show_progress=True)

    image_path_upload.clear(reset_state, outputs=[image_path_upload, chatbot, history], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    # emptyBtn.click(lambda: (None,  chatbot.clear(), gr.State([])), outputs=[image_path_upload, chatbot, history],show_progress=True)
    emptyBtn.click(lambda: (None,  [], []), outputs=[image_path_upload, chatbot, history],show_progress=True)
    print('*'*100)
    print('**************** history2 \n', history)
    print('**************** chatbot2 \n', chatbot)
    # demo.queue().launch(share=args.share, inbrowser=True, server_name='0.0.0.0', server_port=8090)
demo.queue().launch(server_name='0.0.0.0', server_port=8000)

