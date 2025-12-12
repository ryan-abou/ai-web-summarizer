
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# --- minimal fix: set a User-Agent so WebBaseLoader doesn't warn/block ---
os.environ.setdefault(
    "USER_AGENT",
    "ai-web-summarizer/0.1 (+https://github.com/ryan-abou)"
)

# Initialize model and chain
# --- minimal fix: constrain context/output to reduce OOM/crash risk ---
model = ChatOllama(
    model="llama3",
    temperature=0,
    model_kwargs={
        "num_ctx": 4096,     # keep prompt within a modest context window
        "num_predict": 512   # cap output length
    }
)
prompt = ChatPromptTemplate.from_template(
    "Summarize the webpage:\n"
    "1) 4–6 sentence summary\n"
    "2) 5 bullet points\n"
    "3) One takeaway.\n\n"
    "Content:\n{content}"
)
parser = StrOutputParser()
chain = prompt | model | parser

def summarize_url(url: str):
    try:
        docs = WebBaseLoader(url).load()
        if not docs:
            return "No content found."
    except Exception as e:
        return f"Failed to load URL: {e}"

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Combine chunks into one string
    text = ""
    for chunk in chunks:
        if text:
            text += "\n"
        text += chunk.page_content

    # --- minimal fix: hard cap the input size to stay within context/memory ---
    MAX_CHARS = 12000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    # Run the chain
    return chain.invoke({"content": text})

# Call the function
url = "https://ryan-abou.github.io/racism-in-other-wes-moore/"
summary = summarize_url(url)
print(summary)