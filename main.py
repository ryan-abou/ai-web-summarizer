import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3", temperature=0)
prompt = ChatPromptTemplate.from_template(
    "Summarize the webpage:\n"
    "1) 4–6 sentence summary\n"
    "2) 5 bullet points\n"
    "3) One takeaway.\n\n"
    "Content:\n{content}"
)
parser = StrOutputParser()

# Create the chain
chain = prompt | model | parser

# Loads a webpage, chunk the text for long pages, and generate a structured summary.
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

    # Combine chunks
    text = "\n".join(chunk.page_content for chunk in chunks)

    # Run the chain 
    return chain.invoke({"content": text})

url = "https://ryan-abou.github.io/racism-in-other-wes-moore/"
print("\n=== SUMMARY ===\n")
print(summarize_url(url))
