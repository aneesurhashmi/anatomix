import re

def extract_bounding_boxes_and_refs(text, scale=1, return_refs=False, other_model = False):
    """
    Extract [[ref_text_or_plain, [x1,y1,x2,y2]], ...] from a string.

    """
    bboxes, texts = [], []
    last_end = 0
    box_pattern = r"<box>\s*\((.*?)\)\s*</box>"
    ref_pattern = r"<ref>(.*?)</ref>"

    if other_model:
        matches = re.finditer(r"(.*?)(\[\s*[-\d.,\s]+\s*\])", text)
        for m in matches:
            visible = m.group(1).strip()
            nums = re.findall(r"-?\d+(?:\.\d+)?", m.group(2))
            if len(nums) != 4:
                continue
            coords = [float(n) * scale for n in nums]
            ref_text = " ".join(visible.split()) if visible else None
            texts.append(ref_text)
            bboxes.append(coords)
    else:
        # Shared logic for tag-based models
        box_iter = re.finditer(box_pattern, text, flags=re.DOTALL)
        for m in box_iter:
            box_inner = m.group(1)
            nums = re.findall(r"-?\d+(?:\.\d+)?", box_inner)
            if len(nums) != 4:
                last_end = m.end()
                continue

            coords = [float(n) * scale for n in nums]
            context = text[last_end:m.start()]

            # Find the nearest preceding ref text
            ref_matches = re.findall(ref_pattern, context, flags=re.DOTALL)
            if ref_matches:
                ref_text = ref_matches[-1].strip()
            else:
                visible = re.sub(r"<.*?>", "", context, flags=re.DOTALL).strip()
                ref_text = " ".join(visible.split()) if visible else None

            texts.append(ref_text)
            bboxes.append(coords)
            last_end = m.end()
    if return_refs:
        return bboxes, texts
    return bboxes




def plot_bboxes(
    image,
    pred_bboxes,
    gt_bboxes,
    pred_labels=None,
    gt_labels=None,
    pred_color=(255, 99, 71),
    gt_color=(50, 205, 50),
    pred_fill=(255, 99, 71, 60),
    gt_fill=(50, 205, 50, 60),
    width=3,
    show_labels=True,
    font_size=16      # 👈 New parameter
):
    """
    Overlay predicted and ground truth bounding boxes (xyxy) on a PIL image.
    Supports optional per-box labels and adjustable font size.
    """
    img = image.convert("RGBA")

    # Separate overlays for fills and outlines
    fill_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    outline_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(fill_layer)
    outline_draw = ImageDraw.Draw(outline_layer)

    # Load font with adjustable size
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        # print("Excpt")
        font = ImageFont.load_default(font_size)

    def draw_label(draw, x, y, text, color):
        """Draw label with subtle black shadow for visibility."""
        draw.text((x + 1, y + 1), text, fill="black", font=font)
        draw.text((x, y), text, fill=color, font=font)

    # Fill areas first
    for box in pred_bboxes:
        fill_draw.rectangle(box, fill=pred_fill)
    for box in gt_bboxes:
        fill_draw.rectangle(box, fill=gt_fill)

    # Draw outlines and labels
    for i, box in enumerate(pred_bboxes):
        outline_draw.rectangle(box, outline=pred_color, width=width)
        if show_labels:
            x1, y1, _, _ = box
            label = pred_labels[i] if pred_labels and i < len(pred_labels) else "Pred"
            draw_label(outline_draw, x1 + 4, y1 + 4, label, pred_color)

    for i, box in enumerate(gt_bboxes):
        outline_draw.rectangle(box, outline=gt_color, width=width)
        if show_labels:
            x1, y1, _, _ = box
            label = gt_labels[i] if gt_labels and i < len(gt_labels) else "GT"
            draw_label(outline_draw, x1 + 4, y1 + 4, label, gt_color)

    # Blend layers
    combined = Image.alpha_composite(img, fill_layer)
    combined = Image.alpha_composite(combined, outline_layer)

    return combined.convert("RGB")
