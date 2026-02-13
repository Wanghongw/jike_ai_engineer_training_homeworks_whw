import os
from pathlib import Path
from typing import List, Union, Optional

import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from ocr_class import ImageOCRReader


def main():
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    print('api_key:>>> ', api_key)

    Settings.llm = OpenAILike(
        model="qwen-plus",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
        is_chat_model=True
    )
    Settings.embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
        api_key=api_key,
    )

    # 识别出来图片并存入图片列表中
    image_abs_url_path_list = []
    # 自己本地的图片存储目录
    # data_dir = Path(
    #     "C:\\Users\\86157\\Desktop\\github_files\\jike_ai_engineer_training_homeworks_whw\\module3_homework_1\\data_files"
    # )
    data_dir = Path("data_files")
    image_files = list(data_dir.glob('*.png'))
    if not image_files:
        print("没有找到png图片文件!!!")
        return
    for img in image_files:
        print('图片文件名: ', img.name)
        image_abs_url_path_list.append(img.absolute())
    print('图片绝对路径:>> ', image_abs_url_path_list)

    # 初始化 ImageOCRReader 加载图像并生成 Document
    img_reader = ImageOCRReader(lang='en', use_gpu=False)
    documents = img_reader.load_data(image_files)
    print('成功加载文档:>>> ')
    for idx, doc in enumerate(documents, 1):
        print(f'文档 {idx}')
        print(f'Text: {doc.text[:100]} ...')
        # print(f'Metadata: {doc.metadata}')

    # 构建索引并查询
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    # 查询1:
    question1 = '截图中的邮箱是什么'
    print(f'问题1: {question1}')
    resp1 = query_engine.query(question1)
    print(f'回答1: {resp1}')

    # 查询2:
    ques2 = '截图中的代码用的是什么语言？实现的是社么功能？'
    print(f'问题2: {ques2}')  # huohuo@naruto.cc
    resp2 = query_engine.query(ques2)
    print(f'回答2: \n {resp2}')  # 截图中的代码使用的是 Go 语言。实现的功能是冒泡排序（bubble sort），对一个整数切片进行升序排列。


if __name__ == '__main__':
    main()
