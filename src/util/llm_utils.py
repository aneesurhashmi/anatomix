# pip install accelerate
import torch
import re
# try:
#     import sys
#     sys.path.append("/path/to/anatomix/cloned/RaDialog_v2")
#     sys.path.append("/path/to/anatomix/cloned/RaDialog-interactive-radiology-report-generation")

#     from pathlib import Path
#     from PIL import Image
#     import numpy as np
#     from huggingface_hub import snapshot_download

#     from LLAVA_Biovil.llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, remap_to_uint8
#     from LLAVA_Biovil.llava.model.builder import load_pretrained_model
#     from LLAVA_Biovil.llava.conversation import SeparatorStyle, conv_vicuna_v1

#     from LLAVA_Biovil.llava.constants import IMAGE_TOKEN_INDEX
#     from src.utils import create_chest_xray_transform_for_inference, init_chexpert_predictor

# except:
    # print('Radialog imports failed')
# import sys
# sys.path.append('/path/to/anatomix/')

def medgemma_init(model_type = "it", device='auto'):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    # model_type: pt: Pretrained or it: Instruction Tunned (see the paper or HF space for details)
    custom_cache_dir = "/path/to/data/huggingface_ckpts"
    model_id = f"google/medgemma-4b-{model_type}"

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        cache_dir= custom_cache_dir
    )
    processor = AutoProcessor.from_pretrained(
        model_id, 
        cache_dir= custom_cache_dir
        )
    return model, processor


# def maira2_init(cache_dir="", device='auto'):
#     from transformers import AutoModelForCausalLM, AutoProcessor

#     model = AutoModelForCausalLM.from_pretrained(
#         "microsoft/maira-2", 
#         cache_dir= cache_dir,
#         trust_remote_code=True
#         )
#     processor = AutoProcessor.from_pretrained(
#         "microsoft/maira-2", 
#         cache_dir= cache_dir,
#         trust_remote_code=True
#         )

#     model = model.eval()
#     model = model.to(device)

#     return model, processor


# def radialog_init(cache_dir="", device='auto'):
#     repo_id="Chantal/RaDialog-interactive-radiology-report-generation"
#     model_path = snapshot_download(repo_id=repo_id, revision="main", cache_dir=cache_dir)
#     model_path = Path(model_path)


#     tokenizer, model, image_processor, context_len = load_pretrained_model(
#         model_path, 
#         model_base='liuhaotian/llava-v1.5-7b',
#         model_name="llava-v1.5-7b-task-lora_radialog_instruct_llava_biovil_unfrozen_2e-5_5epochs_v5_checkpoint-21000", 
#         load_8bit=False, 
#         load_4bit=False,
#         cache_dir=cache_dir,
#         device=device
#     )
#     clf_ckpt_path = '/path/to/data/huggingface_ckpts/Radialog_other_ckpt/ChexpertClassifier.ckpt'
#     cp_model, cp_class_names, cp_transforms = init_chexpert_predictor(clf_ckpt_path)

#     model.config.tokenizer_padding_side = "left"

#     return {
#         "model": model,
#         "tokenizer":tokenizer,
#         "cp_model":cp_model, 
#         "cp_class_names":cp_class_names, 
#         "cp_transforms":cp_transforms
#     }


# def radvlm_init(cache_dir="", device='auto'):
#     from transformers import AutoProcessor, AutoModelForVision2Seq
#     processor = AutoProcessor.from_pretrained(
#         "KrauthammerLab/RadVLM",
#         cache_dir= cache_dir,
#         local_files_only=True
#         )
#     model = AutoModelForVision2Seq.from_pretrained(
#         "KrauthammerLab/RadVLM",
#         cache_dir= cache_dir,
#         device_map=device,
#         local_files_only=True
#     )
#     return model, processor


# @torch.no_grad()
# def radialog_res(
#     model, 
#     tokenizer,  
#     cp_model, 
#     cp_class_names, 
#     cp_transforms,
#     image, 
#     max_new_tokens=300,
#     prompt=None,
#     report_gen_task=True,
#     ):

#     image = image.convert("L")
#     cp_image = cp_transforms(image)
#     logits = cp_model(cp_image[None].half().cuda())
#     preds_probs = torch.sigmoid(logits)
#     preds = preds_probs > 0.5
#     pred = preds[0].cpu().numpy()
#     findings = cp_class_names[pred].tolist()
#     findings = ', '.join(findings).lower().strip()

#     conv = conv_vicuna_v1.copy()
#     if report_gen_task:
#         PROMPT = f"<image>. Predicted Findings: {findings}. You are to act as a radiologist and write the finding section of a chest x-ray radiology report for this X-ray image and the given predicted findings. Write in the style of a radiologist, write one fluent text without enumeration, be concise and don't provide explanations or reasons."
#     else:
#         PROMPT = prompt
#     conv.append_message("USER", PROMPT)
#     conv.append_message("ASSISTANT", None)
#     text_input = conv.get_prompt()

#     # get the image
#     vis_transforms_biovil = create_chest_xray_transform_for_inference(512, center_crop_size=448)
#     image_tensor = vis_transforms_biovil(image).unsqueeze(0)

#     image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)
#     input_ids = tokenizer_image_token(text_input, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

#     stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
#     stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

#     # generate a report
#     with torch.inference_mode():
#         output_ids = model.generate(
#             input_ids,
#             images=image_tensor,
#             do_sample=False,
#             use_cache=True,
#             max_new_tokens=max_new_tokens,
#             stopping_criteria=[stopping_criteria],
#             pad_token_id=tokenizer.pad_token_id
#         )

#     pred = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().replace("</s>", "")
#     return pred

# @torch.no_grad()
# def maira2_res(model, processor, image, prompt, grounding=False, max_new_tokens=500):
#     # image = Image.open(image_path).convert('RGB')
#     if grounding:
#         if prompt[-1] != ".":
#             prompt += "."
        
#         processed_inputs = processor.format_and_preprocess_phrase_grounding_input(
#             frontal_image=image,
#             phrase=prompt,
#             return_tensors="pt",
#         ).to(model.device)
#     else:
#         processed_inputs = processor.format_and_preprocess_reporting_input(
#                     current_frontal=image,
#                     current_lateral=None,
#                     prior_frontal=None,
#                     indication=None,
#                     technique=None,
#                     comparison=None,
#                     prior_report=None,
#                     return_tensors="pt",
#                     get_grounding=grounding
#                     ).to(model.device)
#     output_decoding = model.generate(
#                     **processed_inputs,
#                     max_new_tokens=max_new_tokens,
#                     use_cache=True
#                 )
#     prompt_length = processed_inputs["input_ids"].shape[-1]
#     decoded_text = processor.decode(output_decoding[0][prompt_length:], skip_special_tokens=True)
#     decoded_text = decoded_text.lstrip()  # Findings generation completions have a single leading space
#     try:
#         generated_text = processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)
#     except Exception as e:
#         print(e)
#         return ""
#     if grounding:
#         t, b = generated_text[0]
#         if b is not None:
#             b = " ,".join([str(i) for i in b])
#             generated_text = f"{t} {b}"
#         else:
#             generated_text = t
#     return generated_text

# @torch.no_grad()
# def medgemma_res(image, prompt, processor, model, max_new_tokens = 200, skip_special_tokens=True):
#     image = [i.convert("RGB") for i in image] if type(image) == list else image.convert("RGB")

#     inputs = processor(
#         text=prompt, images=image, return_tensors="pt"
#     ).to(model.device, dtype=torch.bfloat16)

#     input_len = inputs["input_ids"].shape[-1]

#     with torch.inference_mode():
#         generation = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
#         generation = generation[0][input_len:]

#     decoded = processor.decode(generation, skip_special_tokens=skip_special_tokens)
#     return decoded

# @torch.no_grad()
# def radvlm_res(model, processor, image, prompt, chat_history=None, max_new_tokens=1500):
#     if chat_history is None:
#         chat_history = []

#     # Build the chat history 
#     conversation = []
#     for idx, (user_text, assistant_text) in enumerate(chat_history):
#         if idx == 0:
#             conversation.append({
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": user_text},
#                     {"type": "image"},
#                 ],
#             })
#         else:
#             conversation.append({
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": user_text},
#                 ],
#             })
#         conversation.append({
#             "role": "assistant",
#             "content": [
#                 {"type": "text", "text": assistant_text},
#             ],
#         })

#     # Add the current user prompt
#     if len(chat_history) == 0:
#         # First turn includes the image
#         conversation.append({
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": prompt},
#                 {"type": "image"},
#             ],
#         })
#     else:
#         # Subsequent turns without the image
#         conversation.append({
#             "role": "user",
#             "content": [{"type": "text", "text": prompt}],
#         })

#     # Apply the chat template to create the full prompt
#     full_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

#     # Prepare model inputs
#     inputs = processor(images=image, text=full_prompt, return_tensors="pt", padding=True).to(
#         model.device, torch.float16
#     )

#     # Generate the response
#     with torch.inference_mode():
#         output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

#     # Decode the output
#     full_response = processor.decode(output[0], skip_special_tokens=True)
#     response = re.split(r"(user|assistant)", full_response)[-1].strip()

#     # Update chat history
#     chat_history.append((prompt, response))

#     return response, chat_history


def remove_duplicate_sentences(text):
    sentences = text.split('. ')
    seen = set()
    result = []
    for s in sentences:
        if s.lower() not in seen:
            result.append(s)
            seen.add(s.lower())
    return '. '.join(result)


def postprocess_llm_output(text):
    """
    Cleans and post-processes the raw text output from a large language model.

    Args:
        text: The raw string output from the LLM.

    Returns:
        A cleaned, formatted string.
    """
    # 1. Remove leading/trailing whitespace from the overall text
    cleaned_text = text.strip()

    # 2. Split the text into lines for individual processing
    lines = cleaned_text.splitlines()

    # 3. Use a set to store unique lines and maintain insertion order
    # (using a dictionary as a trick for Python 3.7+ to preserve order)
    # A set is ideal for efficient duplicate checking.
    unique_lines = list(dict.fromkeys(lines))

    # 4. Remove empty lines and strip whitespace from each line
    final_lines = [line.strip() for line in unique_lines if line.strip()]

    # 5. Join the lines back together with a newline character
    final_output = '\n'.join(final_lines)

    # 6. Optional: Further clean up formatting, like removing excessive newlines
    # This regex replaces two or more newlines with a single newline.
    final_output = re.sub(r'\n\s*\n', '\n', final_output)

    return final_output