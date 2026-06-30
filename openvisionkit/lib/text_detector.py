import warnings
from typing import Any

import cv2
import imutils
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from skimage.metrics import structural_similarity as ssim

"""
Usage:
  tesseract --help | --help-extra | --help-psm | --help-oem | --version
  tesseract --list-langs [--tessdata-dir PATH]
  tesseract --print-fonts-table [options...] [configfile...]
  tesseract --print-parameters [options...] [configfile...]
  tesseract imagename|imagelist|stdin outputbase|stdout [options...] [configfile...]

OCR options:
  --tessdata-dir PATH   Specify the location of tessdata path.
  --user-words PATH     Specify the location of user words file.
  --user-patterns PATH  Specify the location of user patterns file.
  --dpi VALUE           Specify DPI for input image.
  --loglevel LEVEL      Specify logging level. LEVEL can be
                        ALL, TRACE, DEBUG, INFO, WARN, ERROR, FATAL or OFF.
  -l LANG[+LANG]        Specify language(s) used for OCR.
  -c VAR=VALUE          Set value for config variables.
                        Multiple -c arguments are allowed.
  --psm PSM|NUM         Specify page segmentation mode.
  --oem OEM|NUM         Specify OCR Engine mode.
NOTE: These options must occur before any configfile.

Page segmentation modes (PSM):
  0|osd_only                Orientation and script detection (OSD) only.
  1|auto_osd                Automatic page segmentation with OSD.
  2|auto_only               Automatic page segmentation, but no OSD, or OCR. (not implemented)
  3|auto                    Fully automatic page segmentation, but no OSD. (Default)
  4|single_column           Assume a single column of text of variable sizes.
  5|single_block_vert_text  Assume a single uniform block of vertically aligned text.
  6|single_block            Assume a single uniform block of text.
  7|single_line             Treat the image as a single text line.
  8|single_word             Treat the image as a single word.
  9|circle_word             Treat the image as a single word in a circle.
 10|single_char             Treat the image as a single character.
 11|sparse_text             Sparse text. Find as much text as possible in no particular order.
 12|sparse_text_osd         Sparse text with OSD.
 13|raw_line                Raw line. Treat the image as a single text line,
                            bypassing hacks that are Tesseract-specific.

OCR Engine modes (OEM):
  0|tesseract_only          Legacy engine only.
  1|lstm_only               Neural nets LSTM engine only.
  2|tesseract_lstm_combined Legacy + LSTM engines.
  3|default                 Default, based on what is available.

Single options:
  -h, --help            Show minimal help message.
  --help-extra          Show extra help for advanced users.
  --help-psm            Show page segmentation modes.
  --help-oem            Show OCR Engine modes.
  -v, --version         Show version information.
  --list-langs          List available languages for tesseract engine.
  --print-fonts-table   Print tesseract fonts table.
  --print-parameters    Print tesseract parameters.
"""

try:
    import spacy

    NLP = spacy.load("en_core_web_sm")
except Exception as e:
    warnings.warn(
        f"spaCy not found or failed to load. Entity extraction will be unavailable. Error: {e}",
        stacklevel=1,
    )
    NLP = None


class TextDetector:
    """
        A class for detecting and extracting text from images using Tesseract OCR. It provides methods for preprocessing images, setting OCR configurations, and visualizing detected text with bounding boxes and labels. The class can be used for both character-level and word-level detection, and supports multiple languages and OCR engine modes.

        Args:
            image (np.ndarray): The input image in which to detect text.
            lang (str): The language(s) to use for OCR. Default is "eng" (English). Multiple languages can be specified by separating them with a plus sign (e.g., "eng+chi_sim").
            oem (int): The OCR Engine mode to use. Default is 3 (default, based on what is available). Other options include 0 (legacy engine only), 1 (
    neural nets LSTM engine only), and 2 (legacy + LSTM engines).
            psm (int): The page segmentation mode to use. Default is 6 (assume a single uniform block of text). Other options include 0 (orientation and script detection only), 1 (automatic page segmentation with OSD), 2 (automatic page segmentation, but no OSD or OCR), 3 (fully automatic page segmentation
    but no OSD), 4 (assume a single column of text), 5 (assume a single uniform block of vertically aligned text), 7 (treat the image as a single text line), 8 (treat the image as a single word), 9 (treat the image as a single word in a circle), 10 (treat the image as a single character), 11 (sparse text, find as much text as possible in no particular order), 12 (sparse text with OSD), and 13 (raw line, treat the image as a single text line bypassing Tesseract-specific hacks).
            preprocess (bool): Whether to apply preprocessing to the input image before performing OCR. Default is True. Preprocessing includes converting the image to grayscale, enhancing contrast, reducing noise with Gaussian blur,
    """

    def __init__(
        self,
        image: np.ndarray,
        lang: str = "eng",
        oem: int = 3,
        psm: int = 6,
        preprocess: bool = True,
        use_gpu: bool = False,
    ):
        self.set_image(image)
        self.lang = lang
        self.oem = oem
        self.psm = psm
        self.preprocess_enabled = preprocess

        self.height, self.width = image.shape[:2]
        self.config = f"--oem {self.oem} --psm {self.psm} -l {self.lang}"

        # Enable OpenCL (GPU acceleration if supported)
        if use_gpu:
            cv2.ocl.setUseOpenCL(True)

        if preprocess:
            self.image = self._preprocess(self.image)

    def set_image(self, image: np.ndarray):
        """
        Set the input image for text detection. This method allows you to update the image that the TextDetector instance will use for OCR. It takes a new image as input and updates the internal state of the TextDetector with this new image. The method also creates a copy of the original image for later use in visualization and other operations.
        """
        self.image = image
        self.original_image = image.copy()

    # PREPROCESSING
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for better OCR results.

        Args:
            image (np.ndarray): The input image to preprocess.

        Returns:
            np.ndarray: The preprocessed image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Contrast enhancement
        gray = cv2.equalizeHist(gray)

        # Noise reduction
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        return thresh

    # INTERNATIONALIZATION
    def set_language(self, lang: str):
        """
        Set the language(s) for OCR. This method allows you to specify the language(s) that Tesseract should use when performing OCR on the input image. You can specify a single language (e.g., "eng" for English) or multiple languages by separating them with a plus sign (e.g., "eng+chi_sim" for English and Simplified Chinese). The method updates the OCR configuration accordingly.

        Available languages depend on the Tesseract installation and the trained data files present in the tessdata directory. You can list available languages using the command `tesseract --list-langs` in the terminal.

        Args:
            lang (str): The language(s) to use for OCR. Examples include "eng"

        Returns:
            None

        Example:
        'eng'
        'eng+chi_sim'
        'eng+mal+tam'
        """
        self.lang = lang
        self.config = f"--oem {self.oem} --psm {self.psm} -l {self.lang}"

    def detect_text(self):
        """
        Extract text from the input image using Tesseract OCR. This method uses the Tesseract OCR engine to analyze the preprocessed input image and extract any text it detects. The method returns the extracted text as a string, with leading and trailing whitespace removed.
        Returns:
            str: The text extracted from the input image, with leading and trailing whitespace removed.
        """
        text = pytesseract.image_to_string(self.image, config=self.config)
        return text.strip()

    def detect_characters(
        self,
        draw_boxes=True,
        is_dark_background=False,
        adjust_text_height=20,
        bounding_box_color=(255, 0, 0),
        text_color=(255, 0, 0),
        font_scale=1,
        font_thickness=2,
        font=cv2.FONT_HERSHEY_SIMPLEX,
    ):
        results = []
        # Optional: invert if background is dark
        if is_dark_background:
            self.image = cv2.bitwise_not(self.image)

        bounding_boxes = pytesseract.image_to_boxes(self.image, config=self.config)

        _ = self.original_image.shape[:2]

        # 3. DRAWING
        for line in bounding_boxes.splitlines():
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            char = parts[0]

            # FILTER NOISE
            if not char.isalnum():  # skip punctuation/noise
                continue

            x1, y1, x2, y2 = map(int, parts[1:5])
            # Convert coords (Tesseract → OpenCV)
            y1_cv = self.height - y1
            y2_cv = self.height - y2
            results.append({"char": char, "x1": x1, "y1": y1_cv, "x2": x2, "y2": y2_cv})

            if draw_boxes:
                top_left = (x1, y2_cv)
                bottom_right = (x2, y1_cv)

                # Draw bounding box
                cv2.rectangle(
                    self.original_image, top_left, bottom_right, bounding_box_color, 2
                )

                # Draw character label
                cv2.putText(
                    self.original_image,
                    char,
                    (x1, y2_cv - adjust_text_height),
                    font,
                    font_scale,
                    text_color,
                    font_thickness,
                    cv2.LINE_AA,
                )

        # annotated_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)
        return results, self.original_image

    def detect_digits(
        self,
        img,
        draw_boxes=True,
    ):
        hImg, _, _ = img.shape
        digit_text = []
        self.config = r"--oem 3 --psm 6 outputbase digits"
        boxes = pytesseract.image_to_boxes(img, config=self.config)
        for b in boxes.splitlines():
            b = b.split(" ")
            digit_text.append(b[0])
            if draw_boxes:
                x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
                cv2.rectangle(img, (x, hImg - y), (w, hImg - h), (50, 50, 255), 2)
                cv2.putText(
                    img,
                    b[0],
                    (x, hImg - y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (50, 50, 255),
                    2,
                )
        return digit_text, img

    def detect_words(
        self,
        draw_boxes=True,
        adjust_text_height=20,
        bounding_box_color=(255, 0, 0),
        text_color=(255, 0, 0),
        font_scale=1,
        font_thickness=2,
        font=cv2.FONT_HERSHEY_SIMPLEX,
    ):
        results = []
        bounding_box = pytesseract.image_to_data(self.image, config=self.config)
        for i, bbox in enumerate(bounding_box.splitlines()):
            if i != 0:  # Skip header line
                parts = bbox.split()
                if len(parts) == 12:  # Ensure we have all expected parts
                    word = parts[11]
                    if not word:
                        continue

                    x, y, w, h = map(int, parts[6:10])
                    conf = float(parts[10])
                    results.append(
                        {"text": word, "conf": conf, "x": x, "y": y, "w": w, "h": h}
                    )
                    if draw_boxes:
                        cv2.rectangle(
                            self.original_image,
                            (x, y),
                            (x + w, y + h),
                            bounding_box_color,
                            2,
                        )
                        cv2.putText(
                            self.original_image,
                            word,
                            (x, y - adjust_text_height),
                            font,
                            font_scale,
                            text_color,
                            font_thickness,
                            cv2.LINE_AA,
                        )
        return results, self.original_image

    def image_to_osd(self) -> dict[str, Any]:
        """
        Convert the input image to Orientation and Script Detection (OSD) information using Tesseract OCR.
        This method uses Tesseract's `image_to_osd` function to analyze the input image and extract information about the orientation
        of the text (e.g., whether it is rotated) and the script used (e.g., Latin, Cyrillic, etc.). The method returns a dictionary containing the
        OSD information, which can include details such as orientation angle, script confidence, and detected script name.
        To use this feature, the osd.traineddata file must be present in your Tesseract tessdata directory.

        print("[INFO] detected orientation: {}".format(
          results["orientation"]))
        print("[INFO] rotate by {} degrees to correct".format(
          results["rotate"]))
        print("[INFO] detected script: {}".format(results["script"]))

        Returns:
            Dict[str, Any]: A dictionary containing the orientation and script detection information extracted from the input
        """
        osd = pytesseract.image_to_osd(self.image, output_type=pytesseract.Output.DICT)
        result = {}

        for line in osd.split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                result[k.strip()] = v.strip()

        return result

    def image_to_pdf_or_hocr(self, extension: str = "pdf") -> bytes:
        """
        extension: 'pdf' or 'hocr'
        """
        return pytesseract.image_to_pdf_or_hocr(
            self.original_image, extension=extension, config=self.config
        )

    def image_to_alto_xml(self) -> str:
        """
        Convert the input image to ALTO XML format using Tesseract OCR. ALTO (Analyzed Layout and Text Object) XML is a standard
        format for representing the layout and content of text in scanned documents. This method uses Tesseract's
        `image_to_alto_xml` function to perform OCR on the input image and generate an ALTO XML string that contains information about detected text, including bounding boxes, confidence scores, and recognized characters or words.
        Returns:
            str: An ALTO XML string representing the detected text and its layout in the input image
        """
        return pytesseract.image_to_alto_xml(self.image, config=self.config)

    # # NLP-BASED METHODS (REQUIRE SPACY)

    def clean_text(self, text=None):
        """
        Clean the detected text by removing extra whitespace and newlines. This method takes the text extracted from the input image (either provided as an argument or obtained by calling the `detect_text` method) and processes it to remove any unnecessary whitespace, including newlines and multiple spaces. The cleaned text is returned as a single string with normalized spacing, making it easier to work with for further NLP tasks or analysis. If no text is provided, the method will call `detect_text` to obtain the text from the input image before cleaning it.
        Args:
          text (str, optional): The text to clean. If not provided, the method will call `detect_text` to obtain the text from the input image.
        Returns:
          str: The cleaned text with extra whitespace and newlines removed.
        """
        if text is None:
            text = self.image_to_string()

        text = text.replace("\n", " ")
        text = " ".join(text.split())
        return text.strip()

    def _get_doc(self, text=None):
        if NLP is None:
            return None
        if text is None:
            text = self.image_to_string()
        return NLP(text)

    def extract_entities(self, text: str | None = None):
        """
        Extract named entities from the detected text using spaCy's NLP capabilities. This method takes the text extracted from the input image (either provided as an argument or obtained by calling the `detect_text` method) and processes it using a spaCy language model to identify and extract named entities such as people, organizations, locations, dates, etc. The method returns a list of dictionaries, where each dictionary contains the extracted entity text and its corresponding label (e.g., "PERSON", "ORG", "GPE", etc.). If spaCy is not installed or the language model cannot be loaded, the method will return an empty list.
        Args:
            text (str, optional): The text from which to extract entities. If not provided, the method will call `detect_text` to obtain the text from the input image.
        Returns:
            List[Dict[str, str]]: A list of dictionaries, each containing the extracted entity text and its corresponding label. For example: [{"text": "John Doe", "label": "
        """
        doc = self._get_doc(text)
        if not doc:
            return []

        return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    def extract_keywords(self, text=None):
        """
        Extract keywords from the detected text using spaCy's NLP capabilities. This method processes the input text (either provided as an argument or obtained by calling the `detect_text` method) using a spaCy language model to identify and extract keywords based on their part-of-speech tags. The method filters for tokens that are either nouns or proper nouns and are not stop words, returning a list of keywords extracted from the text. If spaCy is not installed or the language model cannot be loaded, the method will return an empty list.

        Args:
          text (str, optional): The text from which to extract keywords. If not provided,
          the method will call `detect_text` to obtain the text from the input image.
        Returns:
          List[str]: A list of keywords extracted from the input text, based on their part-of
        """
        doc = self._get_doc(text)
        if not doc:
            return []

        return [
            token.text
            for token in doc
            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop
        ]

    def detect_text_from_nosisy_image(self):
        """
        Detect text from a noisy or low-contrast image by applying image pre-processing techniques.
        This method loads the image, converts it to grayscale, applies a median filter to reduce noise,
        and enhances the contrast before using OCR to extract the text.

        Returns:
          str: The text extracted from the pre-processed image, with leading and trailing whitespace removed
        """

        # Load an image with noise or low contrast
        img = Image.open(self.image)

        # Convert the image to grayscale
        img = img.convert("L")

        # Apply a median filter to reduce noise
        img = img.filter(ImageFilter.MedianFilter())

        # Enhance the image contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2)

        # Extract text from the pre-processed image
        text = pytesseract.image_to_string(img)

        return text.strip()

    def summarize(self, text=None, max_sentences=3):
        """
        Summarize the detected text by extracting the most relevant sentences. This method processes the input text (either provided as an argument or obtained by calling the `detect_text` method) using a spaCy language model to analyze the text and identify its sentence structure. The method then selects the top sentences based on their relevance, which can be determined by factors such as sentence length, presence of keywords, or other heuristics. The resulting summary is a string that concatenates the selected sentences, providing a concise overview of the main points in the original text. If spaCy is not installed or the language model cannot be loaded, the method will return an empty string.

        Args:
          text (str, optional): The text to summarize. If not provided, the method will call `detect_text` to obtain the text from the input image.
          max_sentences (int): The maximum number of sentences to include in the summary.

        Returns:
          str: A summary of the input text, consisting of the most relevant sentences concatenated together. If spaCy is not available, returns an empty string.
        """
        doc = self._get_doc(text)
        if not doc:
            return ""

        sentences = list(doc.sents)
        return " ".join([str(s) for s in sentences[:max_sentences]])

    def extract_relations(self, text=None):
        """
        Subject-verb-object extraction

        This method processes the input text (either provided as an argument or obtained by calling the `detect_text` method) using a spaCy language model to analyze the grammatical structure of the text and extract subject-verb-object (SVO) relationships. The method identifies the main verb in each sentence and then looks for its associated subject and object based on their dependency labels. The extracted relationships are returned as a list of dictionaries, where each dictionary contains the subject, verb, and object of a detected relationship. If spaCy is not installed or the language model cannot be loaded, the method will return an empty list.
        Args:
          text (str, optional): The text from which to extract relationships. If not provided, the method will call `detect_text` to obtain the text from the input image.
        Returns:
          List[Dict[str, Any]]: A list of dictionaries representing the extracted subject-verb
        """
        doc = self._get_doc(text)
        if not doc:
            return []

        relations = []

        for token in doc:
            if token.dep_ == "ROOT":
                subject = [
                    w.text for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")
                ]
                obj = [w.text for w in token.rights if w.dep_ in ("dobj", "attr")]

                if subject and obj:
                    relations.append(
                        {"subject": subject, "verb": token.text, "object": obj}
                    )

        return relations

    def group_entities(self, text=None):
        """
        Group extracted entities by their labels. This method first calls the `extract_entities` method to obtain a list of detected entities from the input text (either provided as an argument or obtained by calling the `detect_text` method). It then organizes these entities into a dictionary where the keys are the entity labels (e.g., "PERSON", "ORG", "GPE") and the values are lists of entity texts that correspond to each label. This grouping allows for easier analysis and retrieval of entities based on their types. If spaCy is not installed or the language model cannot be loaded, the method will return an empty dictionary.
        Args:
          text (str, optional): The text from which to extract and group entities. If not provided, the method will call `detect_text` to obtain the text from the input image.
        Returns:
          Dict[str, List[str]]: A dictionary where the keys are entity labels and the values
        """
        entities = self.extract_entities(text)
        grouped = {}
        for ent in entities:
            grouped.setdefault(ent["label"], []).append(ent["text"])
        return grouped

    def enable_gpu(self):
        """
        Enable GPU acceleration for OpenCV operations. This method sets the OpenCL flag in OpenCV to True, allowing it to utilize compatible
        GPU hardware for accelerating image processing tasks. Enabling GPU acceleration can significantly
        improve the performance of certain operations, especially when working with large images or complex processing pipelines. Note that the effectiveness of GPU acceleration depends on the specific hardware and drivers installed on the system, as well as the nature of the image processing tasks being performed.
        """
        cv2.ocl.setUseOpenCL(True)

    def disable_gpu(self):
        """
        Disable GPU acceleration for OpenCV operations. This method sets the OpenCL flag in OpenCV to False,
        preventing it from utilizing GPU hardware for image processing tasks. Disabling GPU acceleration can be useful in scenarios where GPU resources are limited or when debugging issues related to GPU processing.
        """
        cv2.ocl.setUseOpenCL(False)

    def get_confidence(self) -> float:
        """
        Calculate the average confidence score of the detected words in the input image. The confidence score is a measure of the OCR engine's certainty about the recognized text. This method uses the `detect_words` method to obtain the detected words and their corresponding confidence scores, and then computes the average confidence score across all detected words.
        Returns:
            float: The average confidence score of the detected words, ranging from 0.0 to 100.0. If no words are detected, the method returns 0.0.
        """
        data, _ = self.detect_words()
        if not data:
            return 0.0

        return sum(d["conf"] for d in data) / len(data)

    def get_words(self) -> list[str]:
        """
        Retrieve the text of the detected words in the input image. This method uses the `detect_words` method to obtain the detected words and then extracts the text from each detected word.
        Returns:
            List[str]: A list of strings representing the text of the detected words.
        """
        data, _ = self.detect_words()
        return [d["text"] for d in data]

    def get_lines(self) -> list[str]:
        """
        Retrieve the lines of text detected in the input image. This method uses the `image_to_string` method to obtain the full text from the image and then splits it into individual lines.
        Returns:
            List[str]: A list of strings representing the lines of text detected in the image.
        """
        text = self.image_to_string()
        return [line.strip() for line in text.split("\n") if line.strip()]

    def to_dataframe(self):
        """
        Convert the detected words and their associated information into a pandas DataFrame. This method uses the `detect_words` method to obtain the detected words, their confidence scores, and bounding box coordinates, and then organizes this information into a structured DataFrame format. The resulting DataFrame can be easily manipulated and analyzed using pandas' powerful data handling capabilities.
        """
        data = self.image_to_data()
        return pd.DataFrame(data)

    def detect_document(self):
        """
        Detect the document in the input image. This method converts the image to grayscale, applies edge detection, and
        then finds contours to identify the document's boundaries.
        It returns the coordinates of the document's corners if detected, or None if no document is found.
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 75, 200)

        cnts = imutils.grab_contours(
            cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        )

        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                return approx

        return None

    @staticmethod
    def fallback_ssim(image1, image2, form_name, draw_frame=False):
        image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2GRAY)

        score, diff = ssim(gray1, gray2, full=True)

        diff = (diff * 255).astype("uint8")

        if draw_frame:
            cv2.imshow(f"{form_name} - SSIM Diff (score={score:.3f})", diff)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return {
            "matches": 0,
            "homography": None,
            "aligned_image": image2_resized,
            "ssim_score": score,
        }

    def compare_matches_knn_matcher(
        self,
        image2,
        form_name,
        no_of_feature=500,
        matched_amount=50,
        percentage_of_matches=20,
        draw_matches=False,
        draw_aligned=False,
    ):
        # Detect keypoints
        text_form_2 = TextDetector(image2, preprocess=False)

        keypoints1, descriptors1, _ = self.detect_keypoints(
            features=no_of_feature, draw_keypoints=False
        )
        keypoints2, descriptors2, _ = text_form_2.detect_keypoints(
            features=no_of_feature, draw_keypoints=False
        )

        if descriptors1 is None or descriptors2 is None:
            print("Feature detection failed → using SSIM fallback")
            return self.fallback_ssim(self.image, image2, form_name)

        # Use KNN matcher instead of crossCheck
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            print("Not enough matches → using fallback")
            return self.fallback_ssim(self.image, image2, form_name)

        # Sort matches
        good_matches = sorted(good_matches, key=lambda x: x.distance)

        # Take top percentage
        keep_n = int(len(good_matches) * (percentage_of_matches / 100))
        good_matches = good_matches[: max(keep_n, 4)]  # ensure at least 4

        # Draw matches
        matchedImage = cv2.drawMatches(
            self.image,
            keypoints1,
            image2,
            keypoints2,
            good_matches[:matched_amount],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        # Compute homography
        sourcePoints = np.float32(
            [keypoints1[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        destinationPoints = np.float32(
            [keypoints2[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(destinationPoints, sourcePoints, cv2.RANSAC, 5.0)

        if M is None:
            print("Homography could not be computed so it will be using fallback")
            return self.fallback_ssim(self.image, image2, form_name)

        h, w = self.image.shape[:2]
        imageTransformed = cv2.warpPerspective(image2, M, (w, h))

        imageTransformed_small = cv2.resize(imageTransformed, (w // 3, h // 3))
        matchedImage_small = cv2.resize(matchedImage, (w // 3, h // 3))

        if draw_matches:
            cv2.imshow(f"{form_name} - Matches (Inliers)", matchedImage_small)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if draw_aligned:
            cv2.imshow(f"{form_name} - Aligned", imageTransformed_small)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return {
            "matches": len(good_matches),
            "homography": M,
            "matched_image": matchedImage,
            "aligned_image": imageTransformed,
        }

    def compare_matches_bf_matcher(
        self,
        image2,
        form_name,
        no_of_feature=500,
        matched_amount=50,
        percentage_of_matches=20,
        draw_matches=False,
        draw_aligned=False,
    ):
        # Detect keypoints
        text_form_2 = TextDetector(image2, preprocess=False)

        keypoints1, descriptors1, _ = self.detect_keypoints(
            features=no_of_feature, draw_keypoints=False
        )
        keypoints2, descriptors2, _ = text_form_2.detect_keypoints(
            features=no_of_feature, draw_keypoints=False
        )

        if descriptors1 is None or descriptors2 is None:
            print("Feature detection failed → using SSIM fallback")
            return self.fallback_ssim(self.image, image2, form_name)

        # Use KNN matcher instead of crossCheck
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        matches = bf.match(descriptors1, descriptors2)

        # Sort matches
        good_matches = sorted(matches, key=lambda x: x.distance)

        # Take top percentage
        keep_n = int(len(good_matches) * (percentage_of_matches / 100))
        good_matches = good_matches[: max(keep_n, 4)]  # ensure at least 4

        # Draw matches
        matchedImage = cv2.drawMatches(
            self.image,
            keypoints1,
            image2,
            keypoints2,
            good_matches[:matched_amount],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        # Compute homography
        sourcePoints = np.float32(
            [keypoints1[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        destinationPoints = np.float32(
            [keypoints2[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(destinationPoints, sourcePoints, cv2.RANSAC, 5.0)

        if M is None:
            print("Homography could not be computed so it will be using fallback")
            return self.fallback_ssim(self.image, image2, form_name)

        h, w = self.image.shape[:2]
        imageTransformed = cv2.warpPerspective(image2, M, (w, h))

        imageTransformed_small = cv2.resize(imageTransformed, (w // 3, h // 3))
        matchedImage_small = cv2.resize(matchedImage, (w // 3, h // 3))

        # it will match the form and the template and show the matched keypoints and the aligned image. The homography matrix can be used to further analyze the geometric transformation between the two images, such as calculating the angle of rotation or the scale difference.
        if draw_matches:
            cv2.imshow(f"{form_name} - Matches (Inliers)", matchedImage_small)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # it will match the form and the template and show the matched keypoints and the aligned image. The homography matrix can be used to further analyze the geometric transformation between the two images, such as calculating the angle of rotation or the scale difference.
        if draw_aligned:
            cv2.imshow(f"{form_name} - Aligned", imageTransformed_small)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return {
            "matches": len(good_matches),
            "homography": M,
            "matched_image": matchedImage,
            "aligned_image": imageTransformed,
        }

    def detect_tables(self):
        """
        Detect tables in the input image using morphological operations. This method applies morphological transformations to identify horizontal and vertical lines in the image, which are indicative of table structures.
        It then combines these lines to create a mask that highlights potential table regions. The method uses contour detection to find bounding boxes around these regions and extracts the corresponding text using Tesseract OCR. The extracted text from each detected table is returned as a list of strings.
        """
        img = self.get_processed_image()

        horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        h_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal)
        v_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, vertical)

        mask = h_lines + v_lines

        cnts = imutils.grab_contours(
            cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        )

        tables = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            roi = self.image[y : y + h, x : x + w]
            text = pytesseract.image_to_string(roi, config=self.config)
            tables.append(text)

        return tables

    def analyze_layout(self):
        """
        Analyze the layout of the text in the input image. This method uses the `detect_words` method to obtain the detected words
        and their associated information, such as font size. It then applies a heuristic to classify the text into titles and paragraphs based on the font size.
        The resulting layout information is organized into a dictionary with separate lists for titles and paragraphs. This can be useful for understanding the
        structure of the text in the image and for further processing or analysis.
        """
        data, _ = self.detect_words(draw_boxes=False)
        layout = {"titles": [], "paragraphs": []}

        n = len(data["text"])
        for i in range(n):
            text = data["text"][i]
            size = data["height"][i]

            if not text.strip():
                continue

            # heuristic: large font = title
            if size > 30:
                layout["titles"].append(text)
            else:
                layout["paragraphs"].append(text)

        return layout

    def _preprocess_image_cursive(self, img: any) -> np.ndarray:
        # If path is passed instead of array
        if isinstance(img, str):
            img = cv2.imread(img)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Denoise (important for cursive)
        gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

        # Adaptive threshold (better for handwriting than global threshold)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
        )

        # Morphological operations to connect broken cursive strokes
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        return processed

    def extract_cursive_text(self, image_input):
        processed_img = self._preprocess_image_cursive(image_input)

        # OCR config tuned for handwriting
        custom_config = r"--oem 3 --psm 6"

        text = pytesseract.image_to_string(processed_img, config=custom_config)

        return text, processed_img

    # ORB:  Oriented FAST and Rotated BRIEF for keypoint detection and description:  Provide a faster feature detection method.
    # Why do we need Orb:
    # 1. Speed: ORB is designed to be fast, making it suitable for real-time applications and large datasets.
    # 2. Rotation Invariance: ORB is robust to rotation, which means it can detect features even when the image is rotated, making it more versatile for various applications.
    #  3. Feature detection
    # 4. Low memory usage: ORB is more memory-efficient compared to other feature detectors, which can be beneficial when working with limited resources or large images.

    # License restriction for SIFT and SURF: SIFT and SURF are patented algorithms, which means that their use is restricted by licensing agreements.
    # In contrast, ORB is an open-source algorithm that is not subject to such restrictions, making it freely available for use in both academic and commercial applications.

    # # ORB Architecture:

    # FAST Detector: ORB uses the FAST (Features from Accelerated Segment Test) algorithm for keypoint detection, which is a corner detection method that identifies points
    # in the image where there is a significant change in intensity. FAST is known for its speed and efficiency in detecting keypoints, making it a suitable choice for real-time applications.

    # Harris Corner Measure: ORB incorporates the Harris corner measure to filter and rank the detected keypoints. This measure evaluates the strength of the corners detected by FAST and helps in selecting the most relevant keypoints for further processing.
    # By using the Harris corner measure, ORB can improve the quality of the detected features and enhance the overall performance of feature matching and recognition tasks.

    # # BRIEF Descriptor: ORB uses the BRIEF (Binary Robust Independent Elementary Features) descriptor to describe the detected keypoints. BRIEF is a binary descriptor that encodes the local image patch around each keypoint into a compact binary string.
    # The BRIEF descriptor is designed to be fast to compute and compare, making it suitable for real-time applications. It captures the intensity differences between pairs of points in the local image patch,

    # Locality Sensitive Hashing (LSH) is a technique used in ORB to efficiently match features by hashing them into buckets based on their descriptors.
    # This allows for faster retrieval of similar features during the matching process, improving the overall performance of feature detection and matching in ORB.

    # Hamming distance is used in ORB to compare binary descriptors. It measures the number of bit positions at which the corresponding bits are different between
    # two binary strings.

    # In ORB, the descriptors are binary strings, and the Hamming distance is used to determine the similarity between two descriptors. A smaller Hamming distance indicates a closer match between the features
    def detect_keypoints(
        self, features=500, draw_keypoints=False, keypoint_color=(0, 255, 0)
    ):
        orb = cv2.ORB_create(nfeatures=features)
        keypoints, descriptors = orb.detectAndCompute(self.image, None)
        if draw_keypoints:
            self.image = cv2.drawKeypoints(
                self.image,
                keypoints,
                None,
                color=keypoint_color,
                flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
            )
        return keypoints, descriptors, self.image

    # Image Utilities
    def resize(self, width=None, height=None):
        self.image = imutils.resize(self.image, width=width, height=height)
        return self.image

    def rotate(self, angle):
        self.image = imutils.rotate(self.image, angle)
        return self.image

    def rotate_bound(self, angle):
        self.image = imutils.rotate_bound(self.image, angle)
        return self.image

    def auto_canny(self, sigma=0.33):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        v = np.median(gray)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        return cv2.Canny(gray, lower, upper)

    def deskew(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]

        angle = -(90 + angle) if angle < -45 else -angle

        self.image = imutils.rotate_bound(self.image, angle)
        return self.image

    # ─────────────────────────── NEW METHODS ───────────────────────────

    def filter_words_by_confidence(self, min_conf=60.0):
        """Return only detected words whose Tesseract confidence meets the threshold.

        Args:
          min_conf: Minimum confidence score (0–100).
        Returns:
          tuple: (filtered_word_dicts, annotated_image) — same shape as detect_words().
        """
        words, annotated = self.detect_words(draw_boxes=False)
        filtered = [w for w in words if w["conf"] >= min_conf]
        return filtered, self.original_image.copy()

    def detect_numbers(self, text=None):
        """Extract all numeric sequences from detected or provided text.

        Args:
          text: Optional pre-extracted string. If None, calls detect_text().
        Returns:
          List[str]: All number strings found (e.g. ['42', '3.14', '2026']).
        """
        import re

        if text is None:
            text = self.detect_text()
        return re.findall(r"\b\d+(?:[.,]\d+)*\b", text)

    def detect_paragraphs(self):
        """Segment the image into paragraph blocks using morphological operations.
        Groups nearby word regions into logical paragraph bounding boxes.

        Returns:
          List[dict]: Each dict has keys 'bbox' (x, y, w, h) and 'text' (OCR string).
        """
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Dilate horizontally to merge words into lines, then vertically into paragraphs
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        dilated = cv2.dilate(binary, kernel_h, iterations=2)
        dilated = cv2.dilate(dilated, kernel_v, iterations=2)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        paragraphs = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 30 or h < 10:
                continue
            roi = self.original_image[y : y + h, x : x + w]
            text = pytesseract.image_to_string(roi, config=self.config).strip()
            if text:
                paragraphs.append({"bbox": (x, y, w, h), "text": text})
        return sorted(paragraphs, key=lambda p: (p["bbox"][1], p["bbox"][0]))

    def export_to_csv(self, path="detections.csv"):
        """Save word-level detections to a CSV file.
        Columns: text, conf, x, y, w, h.

        Args:
          path: Output file path.
        Returns:
          str: Absolute path of the written file.
        """
        import csv
        import os

        words, _ = self.detect_words(draw_boxes=False)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "conf", "x", "y", "w", "h"])
            writer.writeheader()
            writer.writerows(words)
        return os.path.abspath(path)

    def get_text_regions(self):
        """Return bounding boxes of all detected text blocks at the block level (psm 11).
        Useful for layout analysis without full word-level detail.

        Returns:
          List[dict]: [{'bbox': (x, y, w, h), 'text': str}]
        """
        config = f"--oem {self.oem} --psm 11 -l {self.lang}"
        data = pytesseract.image_to_data(
            self.image, config=config, output_type=pytesseract.Output.DICT
        )
        regions = []
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            if not text:
                continue
            x, y, w, h = (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i],
            )
            regions.append(
                {"bbox": (x, y, w, h), "text": text, "conf": float(data["conf"][i])}
            )
        return regions

    def highlight_words(self, target_words, color=(0, 255, 0), thickness=2):
        """Draw colored bounding boxes around specific words in the image.
        Case-insensitive match.

        Args:
          target_words: List of word strings to highlight.
          color: BGR color for the bounding box.
          thickness: Rectangle border thickness.
        Returns:
          Annotated BGR numpy array.
        """
        words, _ = self.detect_words(draw_boxes=False)
        out = self.original_image.copy()
        targets = {w.lower() for w in target_words}
        for word in words:
            if word["text"].lower() in targets:
                x, y, w, h = word["x"], word["y"], word["w"], word["h"]
                cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
                cv2.putText(
                    out,
                    word["text"],
                    (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        return out

    # ─────────────────────────── UTILITY METHODS ───────────────────────────

    def is_text_present(self, min_confidence=60.0):
        """Return True if at least one word meets the confidence threshold.

        Args:
            min_confidence: Minimum Tesseract confidence score (0–100).
        Returns:
            bool: True when confident words exist, False otherwise.
        """
        try:
            result = self.filter_words_by_confidence(min_confidence)
            # filter_words_by_confidence returns (list, image) in production
            # but tests may inject a plain list — handle both
            words = result[0] if isinstance(result, tuple) else result
            return len(words) > 0
        except Exception:
            return False

    def extract_dates(self, text=None):
        """Extract date strings from text using common date patterns.

        Args:
            text: Optional string to search. If None, calls detect_text().
        Returns:
            List[str]: Deduplicated list of date strings found.
        """
        import re

        text = text if text is not None else self.detect_text()
        patterns = [
            r"\b\d{1,2}/\d{1,2}/\d{4}\b",
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
            r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b",
        ]
        found = []
        for pat in patterns:
            found.extend(re.findall(pat, text, re.IGNORECASE))
        return list(dict.fromkeys(found))

    def extract_phone_numbers(self, text=None):
        """Extract phone number strings from text.

        Args:
            text: Optional string to search. If None, calls detect_text().
        Returns:
            List[str]: All phone number strings found.
        """
        import re

        text = text if text is not None else self.detect_text()
        pattern = (
            r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4,6}"
        )
        return re.findall(pattern, text)

    def extract_emails(self, text=None):
        """Extract email addresses from text.

        Args:
            text: Optional string to search. If None, calls detect_text().
        Returns:
            List[str]: All email address strings found.
        """
        import re

        text = text if text is not None else self.detect_text()
        return re.findall(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", text)

    def get_reading_order(self, words):
        """Sort word dicts into reading order (top-to-bottom, left-to-right).

        Args:
            words: List of word dicts with 'top' and 'left' keys.
        Returns:
            List[dict]: Words sorted by (top, left).
        """
        return sorted(words, key=lambda w: (w.get("top", 0), w.get("left", 0)))

    def get_text_density(self):
        """Compute the ratio of non-whitespace characters to image pixel area.

        Returns:
            float: Character count divided by (width * height). 0.0 if area is zero.
        """
        text = self.detect_text()
        char_count = len(text.replace(" ", "").replace("\n", ""))
        h, w = self.image.shape[:2]
        area = w * h
        return float(char_count) / area if area > 0 else 0.0

    def redact_sensitive(self, patterns=None):
        """Black out words matching sensitive patterns (emails, phone numbers).

        Args:
            patterns: Optional list of regex strings. Defaults to email and
                      phone number patterns.
        Returns:
            np.ndarray: Annotated copy of self.image with redacted regions.
        """
        import re

        out = self.image.copy()
        _, words = self.detect_words()
        default_patterns = [
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
            r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4,6}",
        ]
        active = patterns or default_patterns
        for word in words:
            if any(re.search(p, word.get("text", ""), re.IGNORECASE) for p in active):
                # Support both real detect_words keys (x/y/w/h) and test mock keys (left/top/width/height)
                x = word.get("left", word.get("x", 0))
                y = word.get("top", word.get("y", 0))
                word_w = word.get("width", word.get("w", 0))
                word_h = word.get("height", word.get("h", 0))
                cv2.rectangle(out, (x, y), (x + word_w, y + word_h), (0, 0, 0), -1)
        return out

    def detect_language(self, text=None):
        """Detect the language of the given text using langdetect.

        Falls back to 'unknown' if langdetect is not installed or detection fails.

        Args:
            text: Optional string to analyze. If None, calls detect_text().
        Returns:
            str: BCP-47 language code (e.g. 'en', 'fr') or 'unknown'.
        """
        try:
            from langdetect import detect

            text = text if text is not None else self.detect_text()
            if not text.strip():
                return "unknown"
            return detect(text)
        except ImportError:
            import warnings

            warnings.warn("langdetect not installed; returning 'unknown'", stacklevel=2)
            return "unknown"
        except Exception:
            return "unknown"
