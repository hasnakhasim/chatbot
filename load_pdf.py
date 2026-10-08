from langchain_community.document_loaders import PyPDFLoader

pdf_path = "data/career_guidance.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print("Number of pages:", len(documents))

for document in documents:
    print(document.page_content[:500])