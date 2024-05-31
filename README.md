# Question-Answering on Private Documents using Chroma with Memory

This project demonstrates how to use LangChain and OpenAI's models to create a question-answering system on private documents. It involves loading documents, chunking data, embedding text, and using a vector database for efficient retrieval. Additionally, it includes memory capabilities to handle conversational contexts.

## Prerequisites

Before running the code, ensure you have the following installed:

- Python 3.8+
- Required libraries as specified in `requirements.txt`

## Setup

1. Clone this repository or download the project files.

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file in the project root directory and add your OpenAI API key:
    ```plaintext
    OPENAI_API_KEY=your_openai_api_key
    ```

## Libraries Used

- `os`
- `dotenv`
- `pypdf`
- `docx2txt`
- `wikipedia`
- `openai`
- `langchain`
- `tiktoken`
- `pinecone-client`
- `chromadb`

## Project Structure

- `project_question_answering_on_private_data_chroma_with_memory.ipynb`: Main notebook with code for loading documents, creating embeddings, storing in vector databases, and performing question-answering with conversational memory.
- `requirements.txt`: Lists all the dependencies needed for the project.

## Steps to Run the Project

1. **Loading Documents**

   Load documents in PDF, DOCX, or TXT formats using LangChain Document Loaders.

    ```python
    from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
    ```

2. **Chunking Data**

   Split the loaded documents into smaller chunks for efficient embedding and retrieval.

    ```python
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    ```

3. **Calculating Cost**

   Calculate the embedding cost based on the number of tokens.

    ```python
    import tiktoken
    ```

4. **Embedding and Uploading to Vector Database (Pinecone or Chroma)**

   Embed the text chunks and upload them to Pinecone or Chroma vector databases.

    ```python
    from langchain_community.vectorstores import Pinecone
    from langchain.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    ```

5. **Asking Questions**

   Perform question-answering using the vector store to retrieve relevant chunks and generate responses using OpenAI's GPT-3.5-turbo model.

    ```python
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI
    ```

6. **Adding Memory (Chat History)**

   Implement conversational memory to handle context across multiple queries.

    ```python
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain
    ```

7. **Using a Custom Prompt**

   Customize the system and user prompts to enhance the interaction, such as translating responses to Spanish.

    ```python
    from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    ```

## Running the Code

1. Load a document:
    ```python
    data = load_document('path_to_your_document.pdf')
    ```

2. Split the document into chunks:
    ```python
    chunks = chunk_data(data, chunk_size=256)
    ```

3. Create a vector store (Chroma):
    ```python
    vector_store = create_embeddings_chroma(chunks)
    ```

4. Ask a question:
    ```python
    q = 'What is Vertex AI Search?'
    answer = ask_and_get_answer(vector_store, q)
    print(answer)
    ```

5. Add conversational memory:
    ```python
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    crc = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
    ```

6. Use a custom prompt:
    ```python
    qa_prompt = ChatPromptTemplate.from_messages(messages)
    crc = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, chain_type='stuff', combine_docs_chain_kwargs={'prompt': qa_prompt })
    ```

## Author

[Your Name]

## License

This project is licensed under the MIT License. See the LICENSE file for details.
