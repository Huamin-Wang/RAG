# 智能文档问答系统：基于RAG架构与大模型的实现

基于检索增强生成（RAG）技术构建的智能文档问答系统，能够从文档库中检索相关信息并通过大模型生成自然语言回答，有效解决大模型"幻觉"问题。

## 🌟 项目特点

- **RAG架构**：结合向量检索与大模型，提升回答准确性
- **多格式支持**：解析DOCX、PPTX、PDF、TXT等常见文档格式
- **语义检索**：基于文本向量化实现语义级别的精准匹配
- **私有部署**：保护文档隐私，适合企业内部知识库场景

## 📦 环境要求

- Python 3.8+
- 系统依赖：
  ```bash
  pip install faiss-cpu numpy python-docx python-pptx sentence-transformers langchain pdfplumber
  ```

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/Huamin-Wang/RAG.git
cd RAG
```

### 2. 准备文档

在项目根目录创建`docs`文件夹，并放入需要问答的文档：

```bash
mkdir docs
```

### 3. 构建知识库

```bash
python main.py
# 选择1. 构建知识库
```

### 4. 开始提问

```bash
python main.py
# 选择2. 提问查询
# 输入问题如"Excel中如何插入墨迹公式"
```

## 🔍 系统架构

系统基于RAG（Retrieval-Augmented Generation）架构，主要包含三个模块：

1. **文档处理模块**：解析文档并分割为文本块
2. **向量检索模块**：将文本转换为向量并构建索引
3. **大模型回答模块**：基于检索结果生成自然语言回答

## 📖 核心功能说明

### 文档解析

支持多种文档格式的文本提取：

```python
def extract_text(path):
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".docx":
        # 解析Word文档
    elif ext == ".pptx":
        # 解析PPT文档
    elif ext == ".pdf":
        # 解析PDF文档
    elif ext == ".txt":
        # 解析文本文件
```

### 文本向量化与检索

使用SentenceTransformer生成文本向量，FAISS构建检索索引：

```python
def embed_text_chunks(text_chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_chunks)
    return embeddings, model

def search_chunks(query, model, chunks, index_path="index.faiss"):
    index = faiss.read_index(index_path)
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k=3)  # 返回最相关3个文本块
    return [chunks[i] for i in I[0]]
```

### 大模型回答生成

将检索结果与问题组合为提示词，调用大模型API：

```python
def generate_answer(context_chunks, query):
    context = "\n".join(context_chunks)
    prompt = f"""你是一个教学助手。根据以下资料内容回答问题：
资料：
{context}

问题：{query}

请用简洁清晰的语言回答："""
    res = DouBao.get_answer(prompt)
    return res
```

## 🛠️ 优化方向

- [ ] 支持更多文档格式（如XLSX、MD）
- [ ] 改进文本分块策略（基于语义分割）
- [ ] 使用更先进的向量模型（如multi-qa-mpnet-base）
- [ ] 增加用户界面（Web/桌面应用）
- [ ] 实现增量更新知识库

## 📝 示例输出

```
请选择操作：
1. 构建知识库
2. 提问查询
3. 退出
请输入数字选择操作：2
请输入你的问题：高级筛选怎么操作
【问题】：高级筛选怎么操作
【回答】：
打开包含数据的Excel文件，点击"数据"选项卡中的"高级筛选"，在打开的高级筛选对话框中设置条件区域等相关参数来进行高级筛选操作。
```

## 📂 项目结构

```
RAG/
├── docs/                # 文档存储目录
├── chunks.txt          # 分割后的文本块
├── index.faiss         # 向量索引文件
├── main.py             # 主程序
├── DouBao.py           # 大模型API接口
└── requirements.txt    # 依赖列表
```

## 📌 注意事项

1. 需要将`DouBao.py`替换为实际可用的大模型API接口
2. 首次使用需先构建知识库
3. 文档分割可能存在语义不完整问题，可根据需求调整`chunk_size`参数
4. 对于大型文档库，建议使用FAISS-GPU版本提升检索效率

## 🎓 相关资源

- 完整教程：https://mp.weixin.qq.com/s/rg1-_HMCxlSdeW548JugwQ
- SentenceTransformer模型：https://www.sbert.net/
- FAISS向量检索库：https://github.com/facebookresearch/faiss

## 🤝 贡献方式

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 提交Pull Request