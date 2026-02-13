from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from typing import List, Union, Optional
from pathlib import Path
import numpy as np
from paddleocr import PaddleOCR


class ImageOCRReader(BaseReader):
    """
    使用 PP-OCR 从图像中提取文本并返回 Document 的 OCR 读取器。

    该类继承自 LlamaIndex 的 BaseReader，可以无缝集成到 LlamaIndex 的数据加载流程中。
    支持从单个或多个图像文件中提取文本，并将结果封装为 Document 对象。

    Attributes:
        lang (str): OCR 识别语言，默认为 'ch'（中文）
        _ocr (PaddleOCR): PaddleOCR 实例，用于执行实际的 OCR 识别
    """

    def __init__(self, lang='ch', use_gpu=False, **kwargs):
        """
        Args:
            lang (str): OCR 语言设置，支持 'ch'（中文）、'en'（英文）、'fr'（法文）等。
                        默认值为 'ch'。
            use_gpu (bool): 是否使用 GPU 加速 OCR 处理。
                           默认值为 False，使用 CPU 处理。
            **kwargs: 其他传递给 PaddleOCR 的额外参数，例如：
                     - det_db_thresh: 检测阈值
                     - det_db_box_thresh: 框阈值
                     - use_angle_cls: 是否使用角度分类器

        Notice:
            为了性能考虑，PaddleOCR 模型在初始化时就会加载，而不是在每次调用 load_data 时加载。
            这样可以避免重复加载模型的开销。
        """
        super().__init__()
        self.lang = lang
        self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, **kwargs)

    def load_data(self, file: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """
        从单个或多个图像文件中提取文本，返回 Document 列表。

        该方法是 BaseReader 的核心方法实现，负责：
        1. 接收图像文件路径（单个或列表）
        2. 使用 PaddleOCR 对每个图像进行 OCR 识别
        3. 将识别结果封装为 Document 对象
        4. 返回所有 Document 对象的列表

        Args:
            file (Union[str, Path, List[Union[str, Path]]]): 图像路径字符串、Path 对象或路径列表。
                                                             支持的图像格式包括 PNG、JPG、JPEG 等。

        Returns:
            List[Document]: 每个图像对应一个 Document 对象的列表。
                           Document 包含以下内容：
                           - text: 识别出的文本内容，按文本块格式化
                           - metadata: 包含图像路径、OCR 模型、语言、文本块数量、平均置信度等元数据

        Example:
            >>> reader = ImageOCRReader(lang='en', use_gpu=False)
            >>> documents = reader.load_data(['image1.png', 'image2.jpg'])
            >>> print(f"Loaded {len(documents)} documents")
            Loaded 2 documents

        Notice:
            - 如果某个图像未检测到任何文本，会打印警告并跳过该图像
            - 每个文本块会标注序号和置信度，格式为：[Text Block N] (conf: X.XX): 文本内容
        """
        if isinstance(file, (str, Path)):
            files = [file]
        else:
            files = file

        documents = []
        for image_path in files:
            image_path_str = str(image_path)

            result = self._ocr.ocr(image_path_str, cls=True)

            if not result or not result[0]:
                print(f"Warning: No text detected in {image_path_str}")
                continue

            text_blocks = []
            confidences = []

            for i, line in enumerate(result[0]):
                text = line[1][0]
                confidence = line[1][1]

                text_blocks.append(f"[Text Block {i + 1}] (conf: {confidence:.2f}): {text}")
                confidences.append(confidence)

            full_text = "\n".join(text_blocks)

            avg_confidence = np.mean(confidences) if confidences else 0.0

            doc = Document(
                text=full_text,
                metadata={
                    "image_path": image_path_str,
                    "ocr_model": "PP-OCRv4",
                    "language": self.lang,
                    "num_text_blocks": len(text_blocks),
                    "avg_confidence": float(avg_confidence)
                }
            )
            documents.append(doc)

        return documents
