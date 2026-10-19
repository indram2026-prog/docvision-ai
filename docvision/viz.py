"""Overlay renderer: colour-coded boxes + labels on the page image."""
from PIL import Image, ImageDraw, ImageFont

COLORS = {"title": (214, 39, 40), "heading": (255, 127, 14), "paragraph": (31, 119, 180),
          "table": (44, 160, 44), "figure": (148, 103, 189), "form_field": (227, 119, 194), "list_item": (23, 190, 207)}


def draw(img: Image.Image, page, show_cells=True) -> Image.Image:
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im, "RGBA")
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        f = ImageFont.load_default()
    for e in page.elements:
        c = COLORS.get(e.type, (0, 0, 0))
        d.rectangle(e.bbox, outline=c + (255,), width=3, fill=c + (28,))
        d.rectangle((e.bbox[0], e.bbox[1] - 20, e.bbox[0] + 8 + 9 * len(e.type), e.bbox[1]), fill=c + (255,))
        d.text((e.bbox[0] + 4, e.bbox[1] - 19), f"{e.order}:{e.type}", fill="white", font=f)
        if show_cells and e.cell_boxes:
            for row in e.cell_boxes:
                for b in row:
                    d.rectangle(b, outline=c + (140,), width=1)
    return im
