import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

def summarize_url(url: str):
    try:
        docs = WebBaseLoader(url).load()
        if not docs:
            return "No content found."
    except Exception as e:
        return f"Failed to load URL: {e}"

    # Combine chunks for long pages
    text = "\n".join(
        RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        .split_documents(docs)[i].page_content
        for i in range(len(docs))
    )

    llm = ChatOllama(model="llama3", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the webpage:\n1) 4–6 sentence summary\n2) 5 bullet points\n3) One takeaway."),
        ("user", "{content}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"content": text})

url = "https://ryan-abou.github.io/racism-in-other-wes-moore/"
print("\n=== SUMMARY ===\n")
print(summarize_url(url))
