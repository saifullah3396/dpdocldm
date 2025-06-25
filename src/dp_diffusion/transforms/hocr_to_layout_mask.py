import ast

import cv2
import numpy as np
import torch

from atria.core.constants import DataKeys
from atria.core.data.data_transforms import DataTransform
from atria.core.utilities.logging import get_logger


import bs4

import gzip

logger = get_logger(__name__)


class HocrToLayoutMask(DataTransform):

    def __init__(
        self,
        hocr_key: str = DataKeys.HOCR,
        image_key: str = DataKeys.IMAGE,
        layout_key: str = DataKeys.LAYOUT_MASK,
        use_filtered_image_mask: bool = False,
        segmentation_level: str = "line",
        threshold_value: float = 0.0,
        is_stored_as_bytes: bool = True,
    ):
        super().__init__()
        self.hocr_key = hocr_key
        self.image_key = image_key
        self.layout_key = layout_key
        self.use_filtered_image_mask = use_filtered_image_mask
        self.segmentation_level = segmentation_level
        self.threshold_value = threshold_value
        self.is_stored_as_bytes = is_stored_as_bytes

    def get_ocr_image_size(self, soup: bs4.BeautifulSoup):
        image_size_str = self.pages(soup)[0]["title"].split("; bbox")[1]
        w, h = map(int, image_size_str[4 : image_size_str.find(";")].split())
        return (w, h)

    def pages(self, soup: bs4.BeautifulSoup):
        return soup.findAll("div", {"class": "ocr_page"})

    def pars(self, soup: bs4.BeautifulSoup):
        return soup.findAll("p", {"class": "ocr_par"})

    def lines(self, soup: bs4.BeautifulSoup):
        return soup.findAll("span", {"class": "ocr_line"})

    def get_par_bboxes(self, soup: bs4.BeautifulSoup):
        pars = []
        w, h = self.get_ocr_image_size(soup)
        for par in self.pars(soup):
            title = par["title"]
            x1, y1, x2, y2 = map(int, title[5:].split())
            bbox = [x1 / w, y1 / h, x2 / w, y2 / h]
            print(bbox)
            pars.append(bbox)
        return pars

    def get_line_bboxes(self, soup: bs4.BeautifulSoup):
        pars = []
        w, h = self.get_ocr_image_size(soup)
        for par in self.lines(soup):
            title = par["title"]
            x1, y1, x2, y2 = map(int, title[5:].split(";")[0].split())
            bbox = [x1 / w, y1 / h, x2 / w, y2 / h]
            pars.append(bbox)
        return pars

    def create_mask_from_hocr(
        self, hocr_string: str, image: torch.Tensor
    ) -> torch.Tensor:
        soup = bs4.BeautifulSoup(hocr_string, features="xml")
        if self.segmentation_level == "block":
            bboxes = self.get_par_bboxes(soup)
        elif self.segmentation_level == "line":
            bboxes = self.get_line_bboxes(soup)
        else:
            raise ValueError(f"Invalid segmentation level: {self.segmentation_level}")
        mask = torch.ones_like(image).bool()
        _, h, w = image.shape
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            x1, y1 = int(x1 * w), int(y1 * h)
            x2, y2 = int(x2 * w), int(y2 * h)
            mask[:, y1:y2, x1:x2] = False
        return mask

    def filter_image(
        self, image: torch.Tensor, mask: torch.Tensor, kernel_size: int = 5
    ) -> torch.Tensor:
        filtered_image = image.clone()

        # remove regions that are in block mask
        filtered_image[~mask] = filtered_image.max()

        # threshold the image to
        filtered_image_np = filtered_image.numpy()
        filtered_image = cv2.adaptiveThreshold(
            filtered_image_np,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            5,
        )
        # erode and dilate the image to remove noise and fill gaps
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        filtered_image = cv2.morphologyEx(filtered_image, cv2.MORPH_OPEN, kernel)
        filtered_image = torch.from_numpy(filtered_image)
        return filtered_image.astype(bool)

    def _apply_transform(self, sample: dict) -> torch.Tensor:
        image = sample[DataKeys.IMAGE]
        hocr_string = sample[DataKeys.HOCR]
        if isinstance(hocr_string, str):
            try:
                hocr_string = ast.literal_eval(
                    hocr_string
                )  # Converts "b'hello'" → b'hello'
            except (SyntaxError, ValueError):
                pass  # It's a normal string, no need to evaluate
        if isinstance(hocr_string, bytes):
            try:
                hocr_string = gzip.decompress(hocr_string).decode()
            except (OSError, gzip.BadGzipFile):
                pass  # Not actually gzipped, keep as-is
        while isinstance(hocr_string, str) and (
            hocr_string.startswith("b'") or hocr_string.startswith('b"')
        ):
            try:
                hocr_string = ast.literal_eval(
                    hocr_string
                )  # Converts "b'hello'" → b'hello'
            except (SyntaxError, ValueError):
                pass  # It's a normal string, no need to evaluate
        block_mask = self.create_mask_from_hocr(hocr_string, image)
        if self.use_filtered_image_mask:
            filtered_image_mask = self.filter_image(image, block_mask)
            mask = block_mask & filtered_image_mask
        else:
            mask = block_mask
        sample[DataKeys.LAYOUT_MASK] = mask.float() * 2 - 1
        return sample
