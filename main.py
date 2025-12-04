
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import WebBaseLoader

def summarize_url(url: str) -> str:
    # 1) Load page (set User-Agent to avoid blocks/warnings)
    loader = WebBaseLoader(
        url,
        header_template={
            "User-Agent": os.getenv("USER_AGENT", "ai-web-summarizer/0.1"),
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    try:
        docs = loader.load()
        if not docs:
            return "No content found at the URL."
    except Exception as e:
        return f"Failed to load URL: {e}"

    # 2) Chunk content
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    # 3) Local LLM via Ollama (ensure server is running & model is pulled)
    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3"),  # allow override via env
        temperature=0,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    # 4) Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a precise summarizer. Read the webpage content and produce:"
         "\n1) A 4–6 sentence summary"
         "\n2) 5 key bullet points"
         "\n3) If applicable, a single-sentence takeaway."),
        ("user", "{content}")
    ])

    # 5) Chain
    chain = prompt | llm | StrOutputParser()

    # 6) Join chunks (simple approach)
    full_text = "\n\n".join([c.page_content for c in chunks]) if chunks else docs[0].page_content

    try:
        summary = chain.invoke({"content": full_text})
    except Exception as e:
        return f"LLM error: {e}"

    return summary

if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    print("\n=== SUMMARY ===\n")
    print(summarize_url(url))
