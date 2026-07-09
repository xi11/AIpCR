import json
import csv
import joblib
import numpy as np

# -----------------------------
# Class mapping
# -----------------------------
color_dict = {
    0: ("background", (255, 165, 0)),
    1: ("neoplastic epithelial", (255, 0, 0)),
    2: ("Inflammatory", (255, 255, 0)),
    3: ("Connective", (0, 255, 0)),
    4: ("Dead", (0, 0, 0)),
    5: ("non-neoplastic epithelial", (0, 0, 255)),
}

# -----------------------------
# Helpers
# -----------------------------
def close_ring(coords):
    if len(coords) >= 3 and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def nucleus_to_geojson_feature(nuc_id, nuc, coord_scale=1.0, skip_background=True):
    cls_id = int(nuc.get("type", -1))
    if skip_background and cls_id == 0:
        return None

    cls_name, rgb = color_dict.get(cls_id, ("unknown", (255, 255, 255)))

    contour = np.asarray(nuc.get("contour", None))
    if contour is None or contour.ndim != 2 or contour.shape[1] != 2 or contour.shape[0] < 3:
        return None

    xs = contour[:, 0].astype(float) * coord_scale
    ys = contour[:, 1].astype(float) * coord_scale

    ring = [[float(x), float(y)] for x, y in zip(xs, ys)]
    ring = close_ring(ring)

    props = {
        "object_type": "nucleus",
        "nucleus_id": str(nuc_id),
        "class_id": cls_id,
        "class_name": cls_name,
        "color_rgb": list(rgb),
    }

    if "prob" in nuc:
        try:
            props["prob"] = float(np.asarray(nuc["prob"]).max())
        except Exception:
            pass

    if "centroid" in nuc:
        c = np.asarray(nuc["centroid"], dtype=float)
        if c.size >= 2:
            props["centroid_x"] = float(c[0] * coord_scale)
            props["centroid_y"] = float(c[1] * coord_scale)

    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": props,
    }


def nucleus_to_csv_row(nuc, coord_scale=1.0, skip_background=True, include_type_name=True):
    cls_id = int(nuc.get("type", -1))
    if skip_background and cls_id == 0:
        return None

    centroid = nuc.get("centroid", None)
    if centroid is None or len(centroid) < 2:
        return None

    c = np.asarray(centroid, dtype=float)
    x = float(c[0] * coord_scale)
    y = float(c[1] * coord_scale)

    row = [x, y, cls_id]
    if include_type_name:
        row.append(color_dict.get(cls_id, ("unknown", None))[0])

    return row


# -----------------------------
# Main exporter
# -----------------------------
def export_hovernet_geojson_and_csv(
    pred_dat_path,
    out_geojson_path,
    out_csv_path,
    coord_scale=1.0,
    skip_background=True,
):
    """
    coord_scale:
        1.0 if HoverNet ran at same resolution as QuPath WSI
        2 / 4 / 8 if inference was downsampled
    """

    wsi_pred = joblib.load(pred_dat_path)
    if not isinstance(wsi_pred, dict):
        raise ValueError(f"Expected dict, got {type(wsi_pred)}")

    geo_features = []
    csv_rows = []

    skipped = 0

    for nuc_id, nuc in wsi_pred.items():
        if not isinstance(nuc, dict):
            continue

        # GeoJSON
        feat = nucleus_to_geojson_feature(
            nuc_id,
            nuc,
            coord_scale=coord_scale,
            skip_background=skip_background,
        )
        if feat is not None:
            geo_features.append(feat)
        else:
            skipped += 1

        # CSV
        row = nucleus_to_csv_row(
            nuc,
            coord_scale=coord_scale,
            skip_background=skip_background,
            include_type_name=True,
        )
        if row is not None:
            csv_rows.append(row)

    # ---- write GeoJSON ----
    geojson = {"type": "FeatureCollection", "features": geo_features}
    with open(out_geojson_path, "w") as f:
        json.dump(geojson, f)

    # ---- write CSV ----
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_hover", "y_hover", "type", "type_name"])
        writer.writerows(csv_rows)

    print(f"GeoJSON saved: {out_geojson_path} ({len(geo_features)} nuclei)")
    print(f"CSV saved:     {out_csv_path} ({len(csv_rows)} nuclei)")
    print(f"Skipped:      {skipped}")


import openslide

def get_wsi_mpp(wsi_path: str) -> float:
    """
    Return microns-per-pixel (mpp) for an SVS/WSI.
    Tries common OpenSlide/Aperio keys and falls back to objective-power heuristics if needed.
    """
    slide = openslide.OpenSlide(wsi_path)
    props = slide.properties

    # 1) Standard OpenSlide keys (most reliable)
    for k in ("openslide.mpp-x", "openslide.mpp-y"):
        if k in props:
            try:
                return float(props[k])
            except Exception:
                pass

    # 2) Aperio key commonly present in SVS
    if "aperio.MPP" in props:
        try:
            return float(props["aperio.MPP"])
        except Exception:
            pass

    # 3) Fallback: infer from objective power if present (approximate)
    #    Common assumption: 40x ≈ 0.25 µm/px, 20x ≈ 0.50 µm/px, 10x ≈ 1.0 µm/px
    for k in ("openslide.objective-power", "aperio.AppMag"):
        if k in props:
            try:
                mag = float(props[k])
                return 10.0 / mag  # 40 -> 0.25, 20 -> 0.5, 10 -> 1.0
            except Exception:
                pass

    raise ValueError(
        f"Could not determine MPP from slide properties for: {wsi_path}\n"
        f"Available property keys: {sorted(list(props.keys()))[:40]} ..."
    )


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    files = ["001_HE_A2_primary", "002_HE_A1_primary", "004_HE_A1_primary", "007_HE_A1_primary", "008_HE_A1_primary", "011_HE_A1_primary",
             "015_HE_A2_primary", "019_HE_A2_primary", "020_HE_A1_Primary", "021_HE_A1_Primary", "024_HE_A1_Primary", "031_HE_A1_Primary",
             "032_HE_A1_Primary", "038_HE_A1_Primary", "040_HE_A1_Primary", "041_HE_A1_Primary", "054_HE_A1_Primary", "057_HE_A1_Primary", 
             "067_HE_A1_Primary", "069_HE_A1_Primary"]
    for file in files:
        wsi_path = f"/rsrch6/home/trans_mol_path/yuan_lab/TIER1/artemis_lei/Discovery/{file}.svs"
        pred_dat = f"/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/hovernet/{file}/0.dat"

        out_geojson = f"/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/hovernet/{file}/nuclei.geojson"
        out_csv = f"/rsrch6/home/trans_mol_path/yuan_lab/TIER2/artemis_lei/discovery/hovernet/{file}/nuclei.csv"

        #to get mpp of wsi
        mpp= get_wsi_mpp(wsi_path)
        print(mpp)
        print(0.25/mpp)

        export_hovernet_geojson_and_csv(
            pred_dat_path=pred_dat,
            out_geojson_path=out_geojson,
            out_csv_path=out_csv,
            coord_scale=0.25/mpp,      # change to 2/4/8 if QuPath alignment is off
            skip_background=True,
        )
