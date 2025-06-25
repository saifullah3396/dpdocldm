import io
from pathlib import Path
import fitz
from atria.core.constants import DataKeys
from atria.core.data.data_transforms import DataTransform
from PIL import Image


class CCPdfPreprocessTransform(DataTransform):
    def __init__(self):
        super().__init__(None)

    def _apply_transform(self, sample):
        pdf_path = Path(sample[DataKeys.PDF_FILE_PATH])
        page_num = sample[DataKeys.PAGE_NUM]
        pdf_document = fitz.open(pdf_path)
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        image = Image.open(io.BytesIO(pix.tobytes()))
        sample[DataKeys.IMAGE_FILE_PATH] = (
            pdf_path.parent / f"{pdf_path.stem}_page_{page_num}.png"
        )
        sample[DataKeys.IMAGE] = image
        sample[DataKeys.IMAGE_WIDTH] = image.width
        sample[DataKeys.IMAGE_HEIGHT] = image.height
        return sample
