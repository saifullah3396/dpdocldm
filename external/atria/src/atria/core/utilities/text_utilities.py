from __future__ import annotations

import ast
from functools import cached_property
from pathlib import Path
from typing import Union

import bs4


class TesseractOCRReader:
    def __init__(self, ocr_file_path: Union[str, Path], conf_threshold: int = 95):
        try:
            self.conf_threshold = conf_threshold
            ocr_file = Path(ocr_file_path)
            if ocr_file.exists() and ocr_file.stat().st_size > 0:
                with open(ocr_file, "r", encoding="utf-8") as f:
                    xml_string = f.read()
                    if xml_string.startswith("b'"):
                        xml_string = ast.literal_eval(xml_string)
                self._soup = bs4.BeautifulSoup(xml_string, features="xml")
            else:
                print(f"Cannot read file: {ocr_file}.")
        except Exception as e:
            print(f"Exception raised while reading ocr data from file {ocr_file}: {e}")
            exit()

    @cached_property
    def pages(self):
        return self._soup.findAll("div", {"class": "ocr_page"})

    @cached_property
    def blocks(self):
        return self._soup.findAll("div", {"class": "ocr_carea"})

    @cached_property
    def words(self):
        return self._soup.findAll("span", {"class": "ocrx_word"})

    def pars_per_block(self, block):
        return block.findAll("p", {"class": "ocr_par"})

    def lines_per_par(self, par):
        return par.findAll("span", {"class": "ocr_line"})

    def words_per_line(self, line):
        return line.findAll("span", {"class": "ocrx_word"})

    @cached_property
    def image_size(self):
        image_size_str = self.pages[0]["title"].split("; bbox")[1]
        w, h = map(int, image_size_str[4 : image_size_str.find(";")].split())
        return (w, h)

    def parse(self, return_confs=True):
        words = []
        word_bboxes = []
        word_angles = []
        confs = []
        w, h = self.image_size
        for word in self.words:
            title = word["title"]
            conf = int(title[title.find(";") + 10 :])
            if word.text.strip() == "" or conf < self.conf_threshold:
                continue

            # get text angle from line title
            textangle = 0
            parent_title = word.parent["title"]
            if "textangle" in parent_title:
                textangle = int(parent_title.split("textangle")[1][1:3])

            x1, y1, x2, y2 = map(int, title[5 : title.find(";")].split())
            words.append(word.text.strip())
            word_bboxes.append([x1 / w, y1 / h, x2 / w, y2 / h])
            word_angles.append(textangle)
            confs.append(conf)
        if return_confs:
            return words, word_bboxes, word_angles, confs
        else:
            return words, word_bboxes, word_angles
