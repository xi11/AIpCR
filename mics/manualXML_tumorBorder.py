import os
from glob import glob
import openslide
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw, ImageColor
Image.MAX_IMAGE_PIXELS = None
#caner_dir = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/Discovery_tumorBorder/caner'
#karina_dir = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/Discovery_tumorBorder/karina'
#lei_dir_old = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/Discovery_tumorBorder/lei'
lei_dir = '/rsrch9/home/plm/idso_fa1_pathology/TIER1/yutong-tnbc-pcr/discovery_pilot25/xml'
src_dir = '/rsrch9/home/plm/idso_fa1_pathology/TIER1/yutong-tnbc-pcr/discovery_pilot25/raw'
dst_dir = '/rsrch9/home/plm/idso_fa1_pathology/TIER2/yutong-tnbc-pcr/discovery_pilot25/tumorBorder_manual'
overlay_dir = os.path.join(dst_dir, 'overlay')
mask_dir = os.path.join(dst_dir, 'mask')
for out_dir in (dst_dir, overlay_dir, mask_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


# Helper function to parse XML and extract vertices
def parse_xml_annotation(xml_file):
    if not os.path.exists(xml_file):
        return []
    tree = ET.parse(xml_file)
    root = tree.getroot()
    vertices_list = []
    for region in root.findall(".//Vertices"):
        vertices = []
        for vertex in region.findall("Vertex"):
            x = float(vertex.attrib['X'])
            y = float(vertex.attrib['Y'])
            vertices.append((x, y))
        vertices_list.append(vertices)
    return vertices_list

# Helper function to draw annotations onto a canvas
def draw_annotations(draw, annotations, color, alpha):
    fill_color = ImageColor.getrgb(color) + (alpha,)
    for vertices in annotations:
        draw.polygon(vertices, outline=color, fill=fill_color)

# Iterate through slides
slides = sorted(glob(os.path.join(src_dir, '*.svs')))
for slide in slides:
    base_name = os.path.basename(slide)[:-4]
    print(base_name)
    svs_path = slide

    #cancer_xml = os.path.join(caner_dir, f"{base_name}.xml")
    #karina_xml = os.path.join(karina_dir, f"{base_name}.xml")
    lei_xml = os.path.join(lei_dir, f"{base_name}.xml")

    # Parse XML annotations
    #cancer_annotations = parse_xml_annotation(cancer_xml)
    #karina_annotations = parse_xml_annotation(karina_xml)
    lei_annotations = parse_xml_annotation(lei_xml)

    # Open the H&E slide
    slide = openslide.OpenSlide(svs_path)
    dimensions = slide.dimensions

    # Create a blank RGBA image for overlay
    overlay = Image.new("RGBA", dimensions, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    # Create a blank RGB image for a binary mask
    mask_image = Image.new("RGB", dimensions, (0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_image)

    # Draw annotations
    #if cancer_annotations:
    #    draw_annotations(draw, cancer_annotations, "red", 64)
    #    draw_annotations(mask_draw, cancer_annotations, "white", 255)
    #if karina_annotations:
    #    draw_annotations(draw, karina_annotations, "blue", 64)
    #    draw_annotations(mask_draw, karina_annotations, "white", 255)
    if lei_annotations:
        draw_annotations(draw, lei_annotations, "yellow", 64)
        for vertices in lei_annotations:
            mask_draw.polygon(vertices, outline=(255, 255, 255), fill=(255, 255, 255))

    # Convert the slide to an RGB image
    slide_image = slide.read_region((0, 0), 0, dimensions).convert("RGB")

    # Composite the slide image with the overlay
    result = Image.alpha_composite(slide_image.convert("RGBA"), overlay)

    # Convert to RGB for JPEG saving
    result_rgb = result.convert("RGB")
    new_width = int(result_rgb.width * 0.5)
    new_height = int(result_rgb.height * 0.5)
    result_rgb = result_rgb.resize((new_width, new_height))

    # Resize and save the mask as RGB binary image
    mask_resized = mask_image.resize((new_width, new_height), resample=Image.NEAREST)
    mask_output_path = os.path.join(mask_dir, f"{base_name}_mask.png")
    mask_resized.save(mask_output_path, "PNG")

    # Save the overlay result
    output_path = os.path.join(overlay_dir, f"{base_name}_overlay.jpg")
    result_rgb.save(output_path, "JPEG", quality=90)


