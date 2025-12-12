import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# Initialize model and chain
model = ChatOllama(model="llama3.2:3b", temperature=0)
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

    # Debug info
    '''
    print(f"[DEBUG] text length: {len(text)}")
    print(f"[DEBUG] chunks count: {len(chunks)}")
    if chunks:
        print(f"[DEBUG] first chunk length: {len(chunks[0].page_content)}")
    '''
    
    # Run the chain
    return chain.invoke({"content": text})

# Call the function
url = "https://ryan-abou.github.io/racism-in-other-wes-moore/"
print(summarize_url(url))

# AI Wrote a decent amount of this code (debug statistics, model, prompt, parser), but I had to make some fixes to get it working. I fully understand my code.