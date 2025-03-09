from typing import List
import pymupdf
import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt")  # Download sentence tokenizer
nltk.download('punkt_tab')


def is_nested(inner,outer):
    """ This method returns True if the inner boundary box is within the outer boundary box in a pdf."""
    return outer[0]<=inner[0] and outer[1]<=inner[1] and outer[2]>=inner[2] and outer[3]>=inner[3]


def find_nested_boxes(page):
    """ This method is used to find all the nested boxes in a page."""
    nested_boxes=set()
    boxes=[]
    for drawing in page.get_drawings():
        boxes.append(tuple(drawing['rect']))

    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i!=j:
                if is_nested(boxes[i],boxes[j]):
                    nested_boxes.add(boxes[j])
    return (nested_boxes,boxes)


def page_content(page):
    """ This method is used to retrieve the page content from a pdf page.
        It clubs the content in a box as a single sentence and return the text."""
    (nested_boxes,total_boxes) = find_nested_boxes(page)
    clean_text_new=""
    clean_text_internal=""
    for rect in nested_boxes:
        if rect[2]-rect[0]>100 and rect[3]-rect[1]>100:
            text=page.get_textbox(rect)
            clean_text =' '.join(text.split())
            clean_text_internal+=clean_text
            clean_text_new+=clean_text+". \n"
    if not clean_text_internal:
        clean_text_new+=page.get_text("text")+". \n"

    return clean_text_new


class FinancialDataProcessor:
    def __init__(self):
        pass

    def read_pdf(self, file_obj) -> str:
        """Extract text from uploaded PDF financial statements."""
        doc = pymupdf.open(stream=file_obj.read(), filetype="pdf")
        text = ""
        for page in doc:
            text+=page_content(page)
        return text

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """ This method is used for split text into overlapping chunks.
            It uses the nltk sentence tokenizer to split the sentences as the nltk sent_tokenize splits on full stop(.) """
        sentences = sent_tokenize(text)  # Split text into sentences
        chunks, current_chunk = [], ""
        max_chunk_size=chunk_size
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence  # Start a new chunk

        if current_chunk:
            chunks_new=current_chunk
            chunks.append(chunks_new.strip())  # Add the last chunk

        return chunks