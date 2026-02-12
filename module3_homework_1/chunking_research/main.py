import os

import pandas as pd
from dotenv import load_dotenv
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.node_parser import (
    SentenceSplitter, TokenTextSplitter, SentenceWindowNodeParser, MarkdownNodeParser
)


def get_eval_template():
    '''
    定义一个评估模型的模版，让 LLM 评估生成回答与标准答案的一致性、准确性和完整性
    '''
    EVAL_TEMPLATE_STR = """
        我们提供了一个标准答案和一个由模型生成的回答。请你判断模型生成的回答在语义上是否与标准答案一致、准确且完整。\n
        请只回答 '是' 或 '否'。
        
        标准答案：
        ----------
        {ground_truth}
        ----------

        模型生成的回答：
        ----------
        {generated_answer}
        ----------
    """
    eval_template = PromptTemplate(template=EVAL_TEMPLATE_STR)

    return eval_template


def evaluate_splitter(splitter, documents, question, ground_truth, splitter_name, eval_template):
    '''
    评估不同的分割器对检索和生成质量的影响。

    Params:
        splitter: 用于切分文档的 LlamaIndex 分割器对象。
        documents: 待处理的文档列表。
        question: 用于查询的问题字符串。
        ground_truth: 问题的标准答案。
        splitter_name: 分割器的名称，用于在结果中标识。
        eval_template: 评估模型的模版
    '''
    print(f"--- 开始评估: {splitter_name} ---")

    # 使用指定分割器切分文档
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"文档被切分成 {len(nodes)} 个节点")

    # 打印前3个节点的信息，帮助理解分割效果
    for i, node in enumerate(nodes[:3]):
        content_preview = node.get_content()[:100].replace("\n", " ")
        print(f"  节点 {i + 1} 预览: {content_preview}...")
        if hasattr(node, 'metadata') and 'file_name' in node.metadata:
            print(f"    来源文件: {node.metadata['file_name']}")

    # 建索引
    index = VectorStoreIndex(nodes)

    # 创建查询引擎，特别处理 SentenceWindowNodeParser，因为它需要一个后处理器
    # Notice: 调整 similarity_top_k 为 2，避免检索过多无关节点
    if isinstance(splitter, SentenceWindowNodeParser):
        query_engine = index.as_query_engine(
            similarity_top_k=2,
            node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window")]
        )
    else:
        query_engine = index.as_query_engine(similarity_top_k=2)

    # 检索上下文
    retrieved_nodes = query_engine.retrieve(question)

    # 打印检索到的节点信息
    print(f"\n检索到 {len(retrieved_nodes)} 个节点:")
    for i, node in enumerate(retrieved_nodes):
        score = node.score if hasattr(node, 'score') else 'N/A'
        file_name = node.metadata.get('file_name', 'Unknown') if hasattr(node, 'metadata') else 'Unknown'
        print(f"  节点 {i + 1} - 相似度: {score if isinstance(score, float) else score}, 来源: {file_name}")

    retrieved_context = '\n\n'.join(node.get_content() for node in retrieved_nodes)

    # 评估: 检索出来的上下文是否包含正确答案
    # Notice 标准答案过长的话可以截取部分
    context_contains_answer = "是" if ground_truth in retrieved_context else "否"

    # 评估：LLM 生成的回答是否准确完整
    response = query_engine.query(question)
    generated_answer = str(response)
    # 使用 LLM 作为评判者来评估答案的准确性
    eval_response = Settings.llm.predict(
        eval_template,
        ground_truth=ground_truth,
        generated_answer=generated_answer
    )
    answer_is_accurate = "是" if "是" in eval_response else "否"

    # 人工评估上下文的冗余度
    print("\n检索到的上下文：")
    print(retrieved_context)
    redundancy_score = input(f"请为【{splitter_name}】的上下文冗余程度打分 (1-5, 1表示最不冗余, 5表示最冗余): ")
    while redundancy_score not in ['1', '2', '3', '4', '5']:
        redundancy_score = input("无效输入。请输入1到5之间的整数: ")

    print(f"--- 完成评估: {splitter_name} ---\n")

    # 返回该分割器的结果
    return {
        "分割器": splitter_name,
        "上下文包含答案": context_contains_answer,
        "回答准确": answer_is_accurate,
        "上下文冗余度(1-5)": int(redundancy_score),
        "生成回答": generated_answer.strip().replace("\n", " ")[:20] + "..."
    }


def print_results_table(evaluate_splitter_results: list[dict]):
    """
    打印所有评估结果的汇总表格
    """
    print("\n--- 最终评估结果对比 ---")
    if evaluate_splitter_results:
        df = pd.DataFrame(evaluate_splitter_results)
        print(df.to_markdown(index=False))
    else:
        print('没有最终的评估结果')


def main():
    '''
    入口函数
    '''
    # 使用本地 .env 文件中的key
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    print(f"loaded api key, {api_key[:5] + '*******'}")

    # 这里用阿里云千问模型
    Settings.llm = OpenAILike(
        model="qwen-plus",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
        is_chat_model=True,
    )

    # 用于使用阿里云 DashScope 的嵌入模型
    Settings.embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
        embed_batch_size=6,
        embed_input_length=8192
    )

    # 加载本地文件
    documents = SimpleDirectoryReader("data_files").load_data()

    # 定义具体问题的标准答案
    question = "鲁迅怎么评价史记的？"
    ground_truth = "史家之绝唱，无韵之离骚"

    report_results = []
    evaluate_template = get_eval_template()

    # Notice 句子切片
    sentence_splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=50,
    )
    se_ret = evaluate_splitter(
        sentence_splitter, documents, question, ground_truth, "SentenceSplitter", evaluate_template
    )
    report_results.append(se_ret)

    # Notice 句子窗口切片
    sentence_window_splitter = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text"
    )
    se_window_ret = evaluate_splitter(
        sentence_window_splitter, documents, question, ground_truth, "SentenceWindowNodeParser", evaluate_template
    )
    report_results.append(se_window_ret)

    # Notice Token切片
    token_splitter = TokenTextSplitter(
        chunk_size=256,
        chunk_overlap=32,
        separator="\n"
    )
    token_ret = evaluate_splitter(
        token_splitter, documents, question, ground_truth, "TokenTextSplitter", evaluate_template
    )
    report_results.append(token_ret)

    # Notice Markdown切片
    markdown_splitter = MarkdownNodeParser()
    md_ret = evaluate_splitter(
        markdown_splitter, documents, question, ground_truth, "MarkdownNodeParser", evaluate_template
    )
    report_results.append(md_ret)

    # 打印最终分析结果
    print_results_table(report_results)


if __name__ == "__main__":
    main()
