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

    # 2) Chunk content (helps with long pages)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    # 3) Local LLM via Ollama (ensure server is running & model is pulled)
    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3"),          # exact tag you pulled
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,                                       # deterministic summaries
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

    # 5) Chain: prompt → llm → parse
    chain = prompt | llm | StrOutputParser()

    # 6) Join chunks (simple approach). For very long pages, consider map–reduce.
    full_text = ""
    for c in chunks:
        page = c.page_content
        full_text += page + "\n"

   # full_text = "\n\n".join([c.page_content for c in chunks]) ???? if chunks else docs[0].page_content

    print(f"[DEBUG] docs count: {len(docs)}")
    #if docs:
    #    print(f"[DEBUG] first doc length: {len(docs[0].page_content)}")

    #print(f"[DEBUG] chunks count: {len(chunks)}")
    #if chunks:
    #    print(f"[DEBUG] first chunk length: {len(chunks[0].page_content)}")

    #print(f"[DEBUG] full_text length: {len(full_text)}")
    #print("[DEBUG] full_text head:\n", full_text[:500])


    summary = chain.invoke({"content": full_text})

    print(summary)

if __name__ == "__main__":
    url = "https://ryan-abou.github.io/racism-in-other-wes-moore/"  # <-- Hardcoded URL
    print("\n=== SUMMARY ===\n")
    summarize_url(url)