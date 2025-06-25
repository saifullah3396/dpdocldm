from typing import List, Tuple

import pandas as pd


def _get_line_bboxes(
    bboxes: List[Tuple[int, int, int, int]]
) -> List[Tuple[int, int, int, int]]:
    x = [bboxes[i][j] for i in range(len(bboxes)) for j in range(0, len(bboxes[i]), 2)]
    y = [bboxes[i][j] for i in range(len(bboxes)) for j in range(1, len(bboxes[i]), 2)]

    x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

    assert x1 >= x0 and y1 >= y0
    bbox = [[x0, y0, x1, y1] for _ in range(len(bboxes))]
    return bbox


def _normalize_bbox(
    bbox: Tuple[int, int, int, int], size: Tuple[int, int]
) -> List[int]:
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def _get_sorted_indices(df: pd.DataFrame) -> pd.DataFrame:
    # Function to determine if two words are on the same line
    def is_same_line(word1, word2):
        return abs(word1["cy"] - word2["cy"]) <= word1["bbox_threshold"]

    # Sort by y0 (top to bottom) and then by x0 (left to right)
    df = df.sort_values(by=["cy", "cx"]).reset_index(drop=True)

    # Group words into lines
    lines = []
    current_line = []
    for i in range(len(df)):
        if i == 0:
            current_line.append(df.iloc[i])
        else:
            prev_word = df.iloc[i - 1]
            current_word = df.iloc[i]
            if is_same_line(prev_word, current_word):
                current_line.append(current_word)
            else:
                lines.append(current_line)
                current_line = [current_word]

    # Add the last line
    if current_line:
        lines.append(current_line)

    # Sort each line by x0 (left to right)
    sorted_lines = [pd.DataFrame(line).sort_values(by="cx") for line in lines]

    # Combine sorted lines into a single DataFrame
    sorted_df = pd.concat(sorted_lines).reset_index(drop=True)

    # Get the sorted indices
    sorted_indices = [int(x) for x in sorted_df["index"].tolist()]

    return sorted_indices


def _sorted_indices_in_reading_order(
    word_bboxes: List[Tuple[int, int, int, int]], bbox_threshold: float = 0.4
) -> dict:
    word_coords = []
    for idx, word_bbox in enumerate(word_bboxes):
        word_coords.append(
            {
                "index": idx,
                "cx": word_bbox[0],
                "cy": word_bbox[1],
                "bbox_threshold": (word_bbox[3] - word_bbox[1]) * bbox_threshold,
            }
        )
    df = pd.DataFrame(word_coords)
    return _get_sorted_indices(df)
