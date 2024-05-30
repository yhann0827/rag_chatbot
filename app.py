import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
prompt_template = PromptTemplate(
    input_variables=["question", "chat_history"],
    template = """
    You are a machine learning teaching assistant for students at the University of Leeds. Your role is to help students understand complex topics related to Gaussian Distribution and Linear Regression, which are part of Unit 2 in their curriculum. The vector database you have access to contains detailed lecture notes and information specifically about these topics.

    When answering the user's question "{question}", please follow these guidelines:

    1. **Relevance and Context**:
        - Ensure that your answer is relevant to the user's question and relates to the conversation history "{chathistory}".
        - Answer the question as a standalone response, providing all necessary information for understanding without assuming prior knowledge from previous interactions.

    2. **Completeness and Clarity**:
        - Provide complete answers that cover all aspects of the question.
        - Write in a clear, concise, and easy-to-understand manner suitable for university students who are learning these concepts.

    3. **Use of Database Content**:
        - Use information from the vector database to provide accurate and detailed responses.
        - If the question is covered in the lecture notes, ensure your answer aligns with the material provided in Unit 2: Gaussian Distribution and Linear Regression.

    4. **Handling Uncertainty and Gaps**:
        - If you are unsure about the question or if more context is needed, prompt the user to provide additional information or clarify their query.
        - If the question is not directly addressed in the retrieved documents and is beyond the scope of Unit 2, inform the user that the question is out of topic.

    5. **Encouraging Further Learning**:
        - Where applicable, suggest further readings or topics that might help the user understand the subject better.
    """

)

def load_pdf():
    text=""
    pdf_path='Machine_Learning_Unit2.pdf'
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings, distance_strategy = DistanceStrategy.MAX_INNER_PRODUCT)
    vectorstore.save_local('faiss_vectorstore')
    return vectorstore

def generate_response(question, chat_history, conversation_chain):
    previous_chat_history = '\n'.join([f"{message['role']}:{message['content']}" for message in chat_history])
    formatted_prompt = prompt_template.format(question=question, chathistory=previous_chat_history)  
    response = conversation_chain.invoke(formatted_prompt)
    return response['answer']

def get_conversation_memory(vectorstore):
    llm = ChatOpenAI(model='gpt-3.5-turbo-0125')
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_message=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever = vectorstore.as_retriever(),
        memory=memory,
        get_chat_history=lambda h : h,
    )
    return conversation_chain

def main():
    load_dotenv()
    st.set_page_config(page_title='Machine Learning Educational Chatbot', page_icon=':open_book:')
    st.title('Machine Learning Chatbot')
    st.header('Machine Learning: Gaussian Distribution and Linear Regression :open_book:')
    st.write("Welcome! I'm here to help you with anything related to Gaussian Distribution and Linear Regression in machine learning. Feel free to ask me questions. :nerd_face:")

    # Load the pdf
    raw_text = load_pdf()
    # Split the text into text chunks
    text_chunks = get_text_chunks(raw_text)
    # Create vectorstore
    vectorstore = get_vectorstore(text_chunks)
    #create conversation chain
    conversation_chain = get_conversation_memory(vectorstore)

    # Check if the messages is null, if null, 
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{'role': 'assistant', 'content':'Hi, how can I help you?'}]
        
    for message in st.session_state.messages:
        st.chat_message(message['role']).write(message['content'])

    prompt = st.chat_input('Enter your prompt here')

    if prompt:
        st.chat_message('user').write(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})
        response = generate_response(prompt, st.session_state.messages, conversation_chain)
        st.chat_message('assistant').write(response)
        st.session_state.messages.append({'role':'assistant', 'content': response})

if __name__=='__main__':
    main()