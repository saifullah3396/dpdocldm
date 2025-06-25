import base64
import dataclasses
import json
import math
import os
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from docile.dataset import Field
from PIL import Image
from tqdm import tqdm


@dataclass(frozen=True)
class FieldWithGroups(Field):
    groups: Optional[Sequence[str]] = None


def get_center_line_clusters(line_item):
    # get centers of text boxes (y-axis only)
    centers = np.array([x.bbox.centroid[1] for x in line_item])
    heights = np.array([x.bbox.height for x in line_item])

    n_bins = len(centers)
    if n_bins < 1:
        return {}

    hist_h, bin_edges_h = np.histogram(heights, bins=n_bins)
    bin_centers_h = bin_edges_h[:-1] + np.diff(bin_edges_h) / 2
    idxs_h = np.where(hist_h)[0]
    heights_cluster_centers = np.unique(bin_centers_h[idxs_h].astype(np.int32))
    heights_cluster_centers.sort()

    # group text boxes by heights
    groups_heights = {}
    for field in line_item:
        g = np.array(
            list(
                map(
                    lambda height: np.abs(field.bbox.height - height),
                    heights_cluster_centers,
                )
            )
        ).argmin()
        gid = heights_cluster_centers[g]
        if gid not in groups_heights:
            groups_heights[gid] = [field]
        else:
            groups_heights[gid].append(field)

    hist, bin_edges = np.histogram(centers, bins=n_bins)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    idxs = np.where(hist)[0]
    y_center_clusters = bin_centers[idxs]
    y_center_clusters.sort()
    line_item_height = y_center_clusters.max() - y_center_clusters.min()

    if line_item_height < heights_cluster_centers[0]:
        # there is probably just 1 cluster
        return {0: y_center_clusters.mean()}
    else:
        #  estimate the number of lines by looking at the cluster centers
        clusters = {}
        cnt = 0
        yc_prev = y_center_clusters[0]
        for yc in y_center_clusters:
            if np.abs(yc_prev - yc) < heights_cluster_centers[0]:
                flag = True
            else:
                flag = False
            if flag:
                if cnt not in clusters:
                    clusters[cnt] = [yc]
                else:
                    clusters[cnt].append(yc)
            else:
                cnt += 1
                clusters[cnt] = [yc]
            yc_prev = yc
        for k, v in clusters.items():
            clusters[k] = np.array(v).mean()
    return clusters


def split_fields_by_text_lines(line_item):
    clusters = get_center_line_clusters(line_item)
    new_line_item = []
    for ft in line_item:
        g = np.array(
            # list(map(lambda y: (ft.bbox.centroid[1] - y) ** 2, clusters.values()))
            list(map(lambda y: np.abs(ft.bbox.to_tuple()[1] - y), clusters.values()))
        ).argmin()
        updated_ft = dataclasses.replace(ft, groups=[g])
        new_line_item.append(updated_ft)
    return new_line_item, clusters


def get_sorted_field_candidates(original_fields):
    fields = []
    # for lid, line_item in original_fields.items():

    # clustering of text boxes in a given line item into individual text lines (stored in fieldlabel.groups)
    # line_item, clusters = split_fields_by_text_lines(line_item)
    line_item, clusters = split_fields_by_text_lines(original_fields)

    # sort text boxes by
    line_item.sort(key=lambda x: x.groups)

    # group by lines:
    groups = {}
    for ft in line_item:
        gid = str(ft.groups)
        if gid not in groups.keys():
            groups[gid] = [ft]
        else:
            groups[gid].append(ft)

    # lid_str = f"{lid:04d}" if lid else "-001"

    for gid, fs in groups.items():
        # sort by x-axis (since we are dealing with a single line)
        fs.sort(key=lambda x: x.bbox.centroid[0])
        for f in fs:
            lid_str = f"{f.line_item_id:04d}" if f.line_item_id else "-001"
            updated_f = dataclasses.replace(
                f,
                # groups = [f"{lid:04d}{int(gid.strip('[]')):>04d}"]
                groups=[f"{lid_str}{int(gid.strip('[]')):>04d}"],
            )
            fields.append(updated_f)
    return fields, clusters


def get_data_from_docile(dataset, overlap_thr=0.5):
    data = []
    metadata = []

    for document in tqdm(dataset, desc=f"Generating data from {dataset}"):
        doc_id = document.docid
        # page_to_table_grids = document.annotation.content["metadata"]["page_to_table_grids"]

        kile_fields = [
            FieldWithGroups.from_dict(field.to_dict())
            for field in document.annotation.fields
        ]
        li_fields = [
            FieldWithGroups.from_dict(field.to_dict())
            for field in document.annotation.li_fields
        ]
        for page in range(document.page_count):
            img = document.page_image(page)
            W, H = img.size
            # resize image to 224 x 224 (as required by LayoutLMv3)
            img = img.resize((224, 224), Image.BICUBIC)
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            # NOTE: to decode:
            # base64_decoded = base64.b64decode(img_str)
            # image = Image.open(io.BytesIO(base64_decoded))

            kile_fields_page = [field for field in kile_fields if field.page == page]
            li_fields_page = [field for field in li_fields if field.page == page]
            kile_fields_page = [
                dataclasses.replace(field, bbox=field.bbox.to_absolute_coords(W, H))
                for field in kile_fields_page
            ]
            li_fields_page = [
                dataclasses.replace(field, bbox=field.bbox.to_absolute_coords(W, H))
                for field in li_fields_page
            ]

            ocr = [
                FieldWithGroups.from_dict(word.to_dict())
                for word in document.ocr.get_all_words(page, snapped=True)
            ]
            ocr = [
                dataclasses.replace(
                    ocr_field,
                    bbox=ocr_field.bbox.to_absolute_coords(W, H),
                    fieldtype=[],
                )
                for ocr_field in ocr
            ]

            # 0. Get table grid
            table_grid = document.annotation.get_table_grid(page)
            tables_bbox = (
                table_grid.bbox.to_absolute_coords(W, H) if table_grid else None
            )

            # 1. Tag ocr fields with fieldtypes from kile_fields + li_fields
            # We sort the kile+lir fields by top coordinate and then for each ocr field we performr
            # binary search to find only the kile+lir fields overlapping vertically.
            # Note: original index is kept to preserve original behaviour.
            kile_li_fields_page = kile_fields_page + li_fields_page
            kile_li_fields_page_sorted = sorted(
                enumerate(kile_li_fields_page),
                key=lambda i_f: i_f[1].bbox.top,
            )
            fields_top_coords = [
                field.bbox.top for _, field in kile_li_fields_page_sorted
            ]
            # Max bottom coordinate is needed to have a sorted array for binary search. This means
            # some extra fields will be included in the found range, causing a very minor slowdown.
            fields_bottom_coords_max = [
                field.bbox.bottom for _, field in kile_li_fields_page_sorted
            ]
            for i in range(1, len(kile_li_fields_page_sorted)):
                fields_bottom_coords_max[i] = max(
                    fields_bottom_coords_max[i],
                    fields_bottom_coords_max[i - 1],
                )
            # Indexes to original kile_li_fields_page array
            fields_idxs = [idx for idx, _ in kile_li_fields_page_sorted]

            updated_ocr = []
            for ocr_field in ocr:
                new_ocr_field = dataclasses.replace(ocr_field, groups="")
                # take only fields with bottom coord after ocr_field.bbox.top
                i_l = bisect_right(fields_bottom_coords_max, ocr_field.bbox.top)
                # take only fields with top coord before ocr_field.bbox.bottom
                i_r = bisect_left(fields_top_coords, ocr_field.bbox.bottom)
                for idx in sorted(fields_idxs[i_l:i_r]):
                    field = kile_li_fields_page[idx]
                    if ocr_field.bbox and field.bbox:
                        if (
                            field.bbox.intersection(ocr_field.bbox).area
                            / ocr_field.bbox.area
                            >= overlap_thr
                        ):
                            if field.fieldtype not in ocr_field.fieldtype:
                                new_ocr_field.fieldtype.append(field.fieldtype)
                            new_ocr_field = dataclasses.replace(
                                new_ocr_field, line_item_id=field.line_item_id
                            )
                updated_ocr.append(new_ocr_field)
            ocr = updated_ocr

            # Re-Order OCR boxes
            sorted_fields, _ = get_sorted_field_candidates(ocr)

            tables_ocr = []
            if tables_bbox:
                for i, field in enumerate(sorted_fields):
                    if (
                        tables_bbox.intersection(field.bbox).area / field.bbox.area
                        >= overlap_thr
                    ):
                        tables_ocr.append((i, field))

            # # 2. Split into individual lines, group by line item id
            # for table_i, table_fields in enumerate(tables_ocr):
            text_lines = {}
            # for field in page_fields:
            for i_field, field in tables_ocr:
                gid = field.groups[0][4:]
                if gid not in text_lines:
                    text_lines[gid] = [(i_field, field)]
                else:
                    text_lines[gid].append((i_field, field))
            # now there should be only 1 line_item_id (or first 04d in groups) per each text_lines
            # we need to merge text_lines, if there are several of them assigned to the same line_item_id
            line_items = {}
            # prev_id = 0 + 1000*table_i
            prev_id = 0 + 1000 * page
            for _, fields in text_lines.items():
                line_item_ids = [
                    f.line_item_id for _i, f in fields if f.line_item_id is not None
                ]
                prev_id = line_item_ids[0] if line_item_ids else prev_id
                if prev_id not in line_items:
                    line_items[prev_id] = fields
                else:
                    line_items[prev_id].extend(fields)
            # 3. Append to data, which will be then used to construct NER Dataset
            for lid, fields in line_items.items():
                if lid > 0:
                    for i_field, field in fields:
                        gid = field.groups[0]
                        new_field = dataclasses.replace(
                            field, line_item_id=lid, groups=[f"{lid:04d}{gid[4:]}"]
                        )
                        sorted_fields[i_field] = new_field

            # append data and metadata
            metadata.append(
                {
                    "i": len(data),
                    "doc_id": doc_id,
                    "page_n": page,
                    "img_b64": img_str,
                    "img_w": W,
                    "img_h": H,
                    # "table_n": table_i,
                    # "row_separators": row_sep[table_i]
                }
            )
            data.append(sorted_fields)
    return data, metadata


def load_metadata(src: Path):
    out = []
    with open(src, "r") as json_file:
        out = json.load(json_file)
    return out


def store_metadata(dest: Path, metadata):
    with open(dest, "w") as json_file:
        json.dump(metadata, json_file)


def load_data(src: Path):
    out = []
    with open(src, "r") as json_file:
        A = json.load(json_file)
    for table_data in A:
        out.append([])
        for field in table_data:
            out[-1].append(FieldWithGroups.from_dict(field))
    return out


def store_data(dest: Path, data):
    out = []
    for table_data in data:
        out.append([])
        for field in table_data:
            out[-1].append(
                {
                    "fieldtype": field.fieldtype if field.fieldtype else "background",
                    "bbox": field.bbox.to_tuple(),
                    "groups": field.groups,
                    "line_item_id": field.line_item_id,
                    "page": field.page,
                    "score": field.score,
                    "text": field.text,
                }
            )
    with open(dest, "w") as json_file:
        json.dump(out, json_file)


def preprocess_dataset(
    docile_dataset,
    overlap_thr,
    preprocessed_dataset_path,
    chunk_size=10000,
):
    if len(docile_dataset) > chunk_size:
        num_chunks = math.ceil(len(docile_dataset) / chunk_size)
        for chunk in range(num_chunks):
            chunk_dataset = docile_dataset[
                chunk * chunk_size : (chunk + 1) * chunk_size
            ]
            chunk_dataset.split_name = (
                f"{docile_dataset.split_name}_chunk_{chunk}_of_{num_chunks}"
            )
            # make sure the chunk is stored to disk
            preprocess_dataset(
                chunk_dataset,
                overlap_thr,
                preprocessed_dataset_path,
                chunk_size,
            )

    if preprocessed_dataset_path:
        dataset_name = docile_dataset.data_paths.name
        preprocessed_path = preprocessed_dataset_path / dataset_name
        print(
            f"Loading preprocessed {docile_dataset.split_name} data from path {preprocessed_path}"
        )
        try:
            data = load_data(
                preprocessed_path
                / f"{docile_dataset.split_name}_multilabel_preprocessed_withImgs.json"
            )
            metadata = load_metadata(
                preprocessed_path
                / f"{docile_dataset.split_name}_multilabel_metadata_withImgs.json"
            )
        except Exception:
            print(
                f"Could not load preprocessed {docile_dataset.split_name}, regenerating."
            )
            data, metadata = get_data_from_docile(
                docile_dataset, overlap_thr=overlap_thr
            )
            print(
                f"Storing preprocessed {docile_dataset.split_name} data to {preprocessed_dataset_path}"
            )
            os.makedirs(preprocessed_dataset_path / dataset_name, exist_ok=True)
            store_data(
                preprocessed_path
                / f"{docile_dataset.split_name}_multilabel_preprocessed_withImgs.json",
                data,
            )
            store_metadata(
                preprocessed_path
                / f"{docile_dataset.split_name}_multilabel_metadata_withImgs.json",
                metadata,
            )
    else:
        data, metadata = get_data_from_docile(docile_dataset, overlap_thr=overlap_thr)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--docile_path",
#         type=Path,
#         default=Path("/app/data/docile/"),
#     )
#     parser.add_argument(
#         "--use_BIO_format",
#         action="store_true",
#         default=True,
#     )
#     parser.add_argument(
#         "--overlap_thr",
#         type=float,
#         default=0.5,
#     )
#     parser.add_argument(
#         "--preprocessed_dataset_path",
#         type=Path,
#         default=None,
#     )
#     args = parser.parse_args()
#     val_docile_dataset = Dataset(
#         "val", args.docile_path, load_annotations=False, load_ocr=False
#     )
#     val_dataset = preprocess_dataset(
#         val_docile_dataset,
#         args.overlap_thr,
#         args.preprocessed_dataset_path,
#     )
#     train_docile_dataset = Dataset(
#         "train", args.docile_path, load_annotations=False, load_ocr=False
#     )
#     train_dataset = preprocess_dataset(
#         train_docile_dataset,
#         args.overlap_thr,
#         args.preprocessed_dataset_path,
#     )
