import os
import faiss
import numpy as np
from docx import Document
from pptx import Presentation
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
import DouBao


# ---------- 文档解析 ----------
def extract_text(path):
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".docx":
        doc = Document(path)
        return '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
    elif ext == ".pptx":
        prs = Presentation(path)
        return '\n'.join(shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text"))
    elif ext == ".pdf":
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    elif ext == ".txt":
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError(f"不支持的文件类型: {ext}")

# ---------- 向量生成 ----------
def embed_text_chunks(text_chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_chunks)
    return embeddings, model

# ---------- 索引构建 ----------
def build_index(vectors, save_path="index.faiss"):
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, save_path)

# ---------- 执行搜索 ----------
def search_chunks(query, model, chunks, index_path="index.faiss"):
    index = faiss.read_index(index_path)
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k=3)
    return [chunks[i] for i in I[0]]

# ---------- 调用 GPT 模型回答 ----------
def generate_answer(context_chunks, query):
    context = "\n".join(context_chunks)
    prompt = f"""你是一个教学助手。根据以下资料内容回答问题：
资料：
{context}

问题：{query}

请用简洁清晰的语言回答："""
    # print(f"外部知识库检索到的相关内容：\n{context}\n")
    res = DouBao.get_answer(prompt)
    return res

# ---------- 主流程：构建知识库 ----------
def build_knowledge():
    all_text = ""
    for filename in os.listdir("docs"):
        path = os.path.join("docs", filename)
        try:
            text = extract_text(path)
            all_text += text + "\n"
            print(f"正在解析：{filename}")
        except Exception as e:
            print(f"跳过 {filename}：{e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(all_text)

    vectors, model = embed_text_chunks(chunks)
    build_index(np.array(vectors))
    with open("chunks.txt", "w", encoding="utf-8") as f:
        f.write("\n\n===CHUNK===\n\n".join(chunks))
    print("✅ 构建完成！已保存索引与文本块。")

# ---------- 主流程：提问查询 ----------
def ask_question(query):
    with open("chunks.txt", "r", encoding="utf-8") as f:
        chunks = f.read().split("\n\n===CHUNK===\n\n")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    results = search_chunks(query, model, chunks)
    answer = generate_answer(results, query)
    print(f"\n【问题】：{query}\n【回答】：\n{answer}")

# ---------- 命令行入口 ----------
if __name__ == "__main__":
    while True:
        print("\n请选择操作：")
        print("1. 构建知识库")
        print("2. 提问查询")
        print("3. 退出")
        choice = input("请输入数字选择操作：")

        if choice == "1":
            build_knowledge()
        elif choice == "2":
            query = input("请输入你的问题：")
            ask_question(query)
        elif choice == "3":
            print("已退出程序。")
            break
        else:
            print("无效选择，请重新输入。")
