import os
import gc
import math
import xml.etree.ElementTree as ET
from glob import glob

import openslide
from PIL import Image, ImageDraw, ImageColor

Image.MAX_IMAGE_PIXELS = None
lei_dir = '/rsrch9/home/plm/idso_fa1_pathology/TIER1/yutong-tnbc-pcr/discovery_pilot25/xml'
src_dir = '/rsrch9/home/plm/idso_fa1_pathology/TIER1/yutong-tnbc-pcr/discovery_pilot25/raw'
dst_dir = '/rsrch9/home/plm/idso_fa1_pathology/TIER2/yutong-tnbc-pcr/discovery_pilot25/tumorBorder_manual'
overlay_dir = os.path.join(dst_dir, 'overlay')
mask_dir = os.path.join(dst_dir, 'mask')
for out_dir in (dst_dir, overlay_dir, mask_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

def parse_xml_annotation(xml_file):
    if not os.path.exists(xml_file):
        return []

    tree = ET.parse(xml_file)
    root = tree.getroot()

    vertices_list = []
    for region in root.findall(".//Vertices"):
        vertices = []
        for vertex in region.findall("Vertex"):
            x = float(vertex.attrib["X"])
            y = float(vertex.attrib["Y"])
            vertices.append((x, y))

        if len(vertices) >= 3:
            vertices_list.append(vertices)

    return vertices_list


def scale_annotations(annotations, scale_x, scale_y):
    scaled = []
    for vertices in annotations:
        scaled.append([(x * scale_x, y * scale_y) for x, y in vertices])
    return scaled


def draw_annotations(draw, annotations, color, alpha=64):
    fill_color = ImageColor.getrgb(color) + (alpha,)
    for vertices in annotations:
        if len(vertices) >= 3:
            draw.polygon(vertices, outline=color, fill=fill_color)


# Important: half-resolution may still be huge.
# Use 8, 16, or 32 for QC overlays.
output_downsample = 8

slides = sorted(glob(os.path.join(src_dir, "*.svs")))

for svs_path in slides:
    base_name = os.path.basename(svs_path)[:-4]

    output_path = os.path.join(overlay_dir, f"{base_name}_overlay.jpg")
    mask_output_path = os.path.join(mask_dir, f"{base_name}_mask.png")

    if os.path.exists(output_path) and os.path.exists(mask_output_path):
        continue

    print(f"Processing {base_name}...")

    lei_xml = os.path.join(lei_dir, f"{base_name}.xml")
    lei_annotations = parse_xml_annotation(lei_xml)

    if not lei_annotations:
        print(f"  No annotation found for {base_name}. Skipping.")
        continue

    slide_obj = openslide.OpenSlide(svs_path)

    try:
        full_w, full_h = slide_obj.dimensions

        # Target output size
        out_w = int(full_w / output_downsample)
        out_h = int(full_h / output_downsample)

        # Use the closest pyramid level instead of reading level 0
        level = slide_obj.get_best_level_for_downsample(output_downsample)
        level_downsample = slide_obj.level_downsamples[level]
        level_w, level_h = slide_obj.level_dimensions[level]

        print(
            f"  Full size: {full_w} x {full_h}; "
            f"using level {level}: {level_w} x {level_h}, "
            f"downsample={level_downsample:.2f}"
        )

        # Read only the lower-resolution pyramid level
        slide_image = slide_obj.read_region(
            (0, 0),
            level,
            (level_w, level_h)
        ).convert("RGB")

        # Scale annotations from level-0 coordinates to selected level coordinates
        scale_x = level_w / full_w
        scale_y = level_h / full_h
        lei_scaled = scale_annotations(lei_annotations, scale_x, scale_y)

        # Create overlay only at the selected pyramid level size
        overlay = Image.new("RGBA", (level_w, level_h), (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        draw_annotations(draw, lei_scaled, "yellow", alpha=64)

        # Use single-channel mask instead of RGB mask
        mask_image = Image.new("L", (level_w, level_h), 0)
        mask_draw = ImageDraw.Draw(mask_image)

        for vertices in lei_scaled:
            if len(vertices) >= 3:
                mask_draw.polygon(vertices, outline=255, fill=255)

        # Composite at low resolution
        result = Image.alpha_composite(
            slide_image.convert("RGBA"),
            overlay
        ).convert("RGB")

        # If selected pyramid level does not exactly match desired output size,
        # resize only the low-resolution image, not the full WSI.
        if (level_w, level_h) != (out_w, out_h):
            result = result.resize((out_w, out_h), resample=Image.Resampling.LANCZOS)
            mask_image = mask_image.resize((out_w, out_h), resample=Image.Resampling.NEAREST)

        result.save(output_path, "JPEG", quality=90)
        mask_image.save(mask_output_path, "PNG")

    finally:
        slide_obj.close()

    # Explicit cleanup helps when looping over many large slides
    del slide_obj
    del slide_image
    del overlay
    del draw
    del mask_image
    del mask_draw
    del result

    gc.collect()