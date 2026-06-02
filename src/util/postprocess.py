import re
from PIL import Image, ImageDraw, ImageFont


def extract_bounding_boxes_and_refs(text, scale=1):
    """Parse bounding boxes from model output text.

    Accepts coordinates in a range of formats (parenthesised or not, separated by
    commas, semicolons, or spaces). Partial matches are silently skipped; only
    complete 4-number tuples are returned.

    Args:
        text: Raw model output string that may contain bounding box coordinates.
        scale: Multiply all coordinate values by this factor (useful for converting
               from normalised to pixel space). Defaults to 1 (no scaling).

    Returns:
        List of (xmin, ymin, xmax, ymax) tuples.
    """
    if "<|" in text:
        text = text.replace("|", "")

    pattern = re.compile(
        r"""
        \(?\s*                # optional opening parenthesis
        (-?\d+(?:\.\d+)?)\s* # xmin
        [,; ]*\s*             # separator
        (-?\d+(?:\.\d+)?)\s* # ymin
        \)?\s*[,; ]*\s*\(?   # optional closing / opening parenthesis
        (-?\d+(?:\.\d+)?)?\s* # xmax (optional for partial matches)
        [,; ]*\s*             # separator
        (-?\d+(?:\.\d+)?)?   # ymax (optional)
        \)?
        """,
        re.VERBOSE,
    )

    boxes = []
    for match in pattern.findall(text):
        nums = [float(x) for x in match if x != ""]
        if len(nums) == 4:
            if scale != 1:
                nums = [n * scale for n in nums]
            boxes.append(tuple(nums))

    return boxes


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
    font_size=16,
):
    """Overlay predicted and ground-truth bounding boxes on a PIL image.

    Ground-truth boxes are drawn first; predicted boxes are drawn on top so they
    remain visible. Separate RGBA layers are used for fills and outlines to avoid
    fill opacity affecting the outline rendering.

    Args:
        image: PIL Image to annotate.
        pred_bboxes: List of (x1, y1, x2, y2) predicted boxes.
        gt_bboxes: List of (x1, y1, x2, y2) ground-truth boxes.
        pred_labels / gt_labels: Optional label strings for each box.
        pred_color / gt_color: Outline colour as an RGB tuple.
        pred_fill / gt_fill: Fill colour as an RGBA tuple.
        width: Outline width in pixels.
        show_labels: Draw label text inside each box.
        font_size: Font size for label text.

    Returns:
        Annotated RGB PIL Image.
    """
    img = image.convert("RGBA")
    fill_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    outline_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(fill_layer)
    outline_draw = ImageDraw.Draw(outline_layer)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    def draw_label(draw, x, y, text, color):
        draw.text((x + 1, y + 1), text, fill="black", font=font)
        draw.text((x, y), text, fill=color, font=font)

    for box in gt_bboxes:
        fill_draw.rectangle(box, fill=gt_fill)
    for box in pred_bboxes:
        fill_draw.rectangle(box, fill=pred_fill)

    for i, box in enumerate(gt_bboxes):
        outline_draw.rectangle(box, outline=gt_color, width=width)
        if show_labels:
            x1, y1, _, _ = box
            label = gt_labels[i] if gt_labels and i < len(gt_labels) else "GT"
            draw_label(outline_draw, x1 + 4, y1 + 4, label, gt_color)

    for i, box in enumerate(pred_bboxes):
        outline_draw.rectangle(box, outline=pred_color, width=width)
        if show_labels:
            x1, y1, _, _ = box
            label = pred_labels[i] if pred_labels and i < len(pred_labels) else "Pred"
            draw_label(outline_draw, x1 + 4, y1 + 4, label, pred_color)

    combined = Image.alpha_composite(img, fill_layer)
    combined = Image.alpha_composite(combined, outline_layer)
    return combined.convert("RGB")
