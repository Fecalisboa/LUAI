import base64
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from schema import TransformType, EmbeddingTypes, IndexerType, BotType
import os
from langchain import FAISS, OpenAI, HuggingFaceHub, Cohere, PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, CohereEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, NLTKTextSplitter, SpacyTextSplitter
from langchain.vectorstores import Chroma, ElasticVectorSearch
from pypdf import PdfReader

class QnASystem:

    def read_and_load_pdf(self, f_data):
        pdf_data = PdfReader(f_data)
        documents = []
        for idx, page in enumerate(pdf_data.pages):
            documents.append(Document(page_content=page.extract_text(), metadata={"page_no": idx, "source": f_data.name}))
        self.documents = documents

    def document_transformer(self, transform_type: TransformType):
        match transform_type:
            case TransformType.CharacterTransform:
                t_type = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
            case TransformType.RecursiveTransform:
                t_type = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
            case TransformType.NLTKTransform:
                t_type = NLTKTextSplitter()
            case TransformType.SpacyTransform:
                t_type = SpacyTextSplitter()
            case _:
                raise IndexError("Invalid Transformer Type")
        self.transformed_documents = t_type.split_documents(documents=self.documents)

    def generate_embeddings(self, embedding_type: EmbeddingTypes = EmbeddingTypes.OPENAI, indexer_type: IndexerType = IndexerType.FAISS, **kwargs):
        temperature = kwargs.get("temperature", 0)
        max_tokens = kwargs.get("max_tokens", 512)
        match embedding_type:
            case EmbeddingTypes.OPENAI:
                os.environ["OPENAI_API_KEY"] = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
                embeddings = OpenAIEmbeddings()
                llm = OpenAI(temperature=temperature, max_tokens=max_tokens)
            case EmbeddingTypes.HUGGING_FACE:
                embeddings = HuggingFaceEmbeddings(model_name=kwargs.get("model_name"))
                llm = HuggingFaceHub(repo_id=kwargs.get("model_name"), model_kwargs={"temperature": temperature, "max_tokens": max_tokens})
            case EmbeddingTypes.COHERE:
                os.environ["COHERE_API_KEY"] = kwargs.get("OGY2ZCgZ4351TM0pXzRNeJLpw6o9GhyfWA3r05eW") or os.getenv("COHERE_API_KEY")
                embeddings = CohereEmbeddings(model=kwargs.get("model_name"), cohere_api_key=kwargs.get("api_key"))
                llm = Cohere(model=kwargs.get("model_name"), cohere_api_key=kwargs.get("OGY2ZCgZ4351TM0pXzRNeJLpw6o9GhyfWA3r05eW"), model_kwargs={"temperature": temperature, "max_tokens": max_tokens})
            case _:
                raise IndexError("Invalid Embedding Type")
        match indexer_type:
            case IndexerType.FAISS:
                indexer = FAISS
            case IndexerType.CHROMA:
                indexer = Chroma()
            case IndexerType.ELASTICSEARCH:
                indexer = ElasticVectorSearch(elasticsearch_url=kwargs.get("elasticsearch_url"))
            case _:
                raise IndexError("Invalid Indexer Function")
        self.llm = llm
        self.indexer = indexer
        self.vector_store = indexer.from_documents(documents=self.transformed_documents, embedding=embeddings)

    def get_retriever(self, search_type="similarity", top_k=5, **kwargs):
        retriever = self.vector_store.as_retriever(search_type=search_type, search_kwargs={"k": top_k})
        self.retriever = retriever

    def get_prompt(self, bot_type: BotType, **kwargs):
        match bot_type:
            case BotType.qna:
                prompt = """
                You are a smart and helpful AI assistant, who answers the question given context
                {context}
                Question: {question}
                """
            case BotType.conversational:
                prompt = """
                Given the following conversation and a follow-up question, 
                rephrase the follow-up question to be a standalone question, in its original language.
                \nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:
                """
        return PromptTemplate(input_variables=["context", "question", "chat_history"], template=prompt)

    def build_qa(self, qa_type: BotType, chain_type="stuff", return_documents: bool = True, **kwargs):
        match qa_type:
            case BotType.qna:
                self.chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.retriever, chain_type=chain_type, return_source_documents=return_documents, verbose=True)
            case BotType.conversational:
                self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
                self.chain = ConversationalRetrievalChain.from_llm(llm=self.llm, retriever=self.retriever, chain_type=chain_type, return_source_documents=return_documents, memory=self.memory, verbose=True)
            case _:
                raise IndexError("Invalid QA Type")

    def ask_question(self, query):
        if type(self.chain) == RetrievalQA:
            data = {"query": query}
        else:
            data = {"question": query}
        return self.chain(data)

    def build_chain(self, transform_type, embedding_type, indexer_type, **kwargs):
        if hasattr(self, "llm"):
            return self.chain
        self.document_transformer(transform_type)
        self.generate_embeddings(embedding_type=embedding_type, indexer_type=indexer_type, **kwargs)
        self.get_retriever(**kwargs)
        qa = self.build_qa(qa_type=kwargs.get("bot_type"), **kwargs)
        return qa

kwargs = {}
source_docs = []
st.set_page_config(page_title="PDFChat - An LLM-powered experimentation app")

if "qna_system" not in st.session_state:
    st.session_state.qna_system = QnASystem()

def show_pdf(f):
    f.seek(0)
    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def model_settings():
    kwargs["temperature"] = st.slider("Temperature", max_value=1.0, min_value=0.0)
    kwargs["max_tokens"] = st.number_input("Max Token", min_value=0, value=512)

st.title("PDF Question and Answering")

tab1, tab2, tab3 = st.tabs(["Upload and Ingest PDF", "Ask", "Show PDF"])

with st.sidebar:
    st.header("Advance Setting ⚙️")
    require_pdf = st.checkbox("Show PDF", value=1)
    st.markdown('---')
    kwargs["bot_type"] = st.selectbox("Bot Type", options=BotType)
    st.markdown("---")
    st.text("Model Parameters")
    kwargs["return_documents"] = st.checkbox("Require Source Documents", value=True)
    text_transform = st.selectbox("Text Transformer", options=TransformType)
    st.markdown("---")
    selected_model = st.selectbox("Select Model", options=EmbeddingTypes)

    # Inserir a chave da API diretamente no código
    match selected_model:
        case EmbeddingTypes.OPENAI:
            api_key = "sua_openai_api_key_aqui"
            model_settings()
        case EmbeddingTypes.HUGGING_FACE:
            api_key = "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"
            kwargs["model_name"] = st.selectbox("Choose Model", options=["google/flan-t5-xxl"])
            model_settings()
        case EmbeddingTypes.COHERE:
            api_key = "OGY2ZCgZ4351TM0pXzRNeJLpw6o9GhyfWA3r05eW"
            model_settings()
        case _:
            api_key = None
    kwargs["api_key"] = api_key
    st.markdown("---")

    vector_indexer = st.selectbox("Vector Indexer", options=IndexerType)
    match vector_indexer:
        case IndexerType.ELASTICSEARCH:
            kwargs["elasticsearch_url"] = st.text_input("Elastic Search URL: ")
            if not kwargs.get("elasticsearch_url"):
                st.warning("Please enter your elastic search url", icon='⚠')
            kwargs["elasticsearch_index"] = st.text_input("Elastic Search Index: ")
            if not kwargs.get("elasticsearch_index"):
                st.warning("Please enter your elastic search index", icon='⚠')

    st.markdown("---")
    st.text("Chain Settings")
    kwargs["chain_type"] = st.selectbox("Chain Type", options=["stuff", "map_reduce"])
    kwargs["search_type"] = st.selectbox("Search Type", options=["similarity"])
    st.markdown("---")

with tab1:
    uploaded_file = st.file_uploader("Upload and Ingest PDF 🚀", type="pdf")
    if uploaded_file:
        with st.spinner("Uploading and Ingesting"):
            documents = st.session_state.qna_system.read_and_load_pdf(uploaded_file)
            if selected_model == EmbeddingTypes.NA:
                st.warning("Please select the model", icon='⚠')
            else:
                st.session_state.qna_system.build_chain(transform_type=text_transform, embedding_type=selected_model, indexer_type=vector_indexer, **kwargs)

def generate_response(prompt):
    if prompt and uploaded_file:
        response = st.session_state.qna_system.ask_question(prompt)
        return response.get("answer", response.get("result", "")), response.get("source_documents")
    return "", []

with tab2:
    if not uploaded_file:
        st.warning("Please upload PDF", icon='⚠')
    else:
        match kwargs["bot_type"]:
            case BotType.qna:
                with st.container():
                    with st.form('my_form'):
                        text = st.text_area("", placeholder='Ask me...')
                        submitted = st.form_submit_button('Submit')
                        if text:
                            st.write(f"Question:\n{text}")
                            response, source_docs = generate_response(text)
                            st.write(response)
            case BotType.conversational:
                # Generate empty lists for generated and past.
                ## generated stores AI generated responses
                if 'generated' not in st.session_state:
                    st.session_state['generated'] = ["Hi! I'm PDF Assistant 🤖, How may I help you?"]
                ## past stores User's questions
                if 'past' not in st.session_state:
                    st.session_state['past'] = ['Hi!']

                input_container = st.container()
                colored_header(label='', description='', color_name='blue-30')
                response_container = st.container()
                response = ""

                def get_text():
                    input_text = st.text_input("You: ", "", key="input")
                    return input_text

                with input_container:
                    user_input = get_text()
                    if st.button("Clear"):
                        st.session_state.generated.clear()
                        st.session_state.past.clear()

                with response_container:
                    if user_input:
                        response, source_docs = generate_response(user_input)
                        st.session_state.past.append(user_input)
                        st.session_state.generated.append(response)

                    if st.session_state['generated']:
                        for i in range(len(st.session_state['generated'])):
                            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                            message(st.session_state["generated"][i], key=str(i))

        require_document = st.container()
        if kwargs["return_documents"]:
            with require_document:
                with st.expander("Related Documents", expanded=False):
                    for source in source_docs:
                        metadata = source.metadata
                        st.write("{source} - {page_no}".format(source=metadata.get("source"), page_no=metadata.get("page_no")))
                        st.write(source.page_content)
                        st.markdown("---")

with tab3:
    if require_pdf and uploaded_file:
        show_pdf(uploaded_file)
    elif uploaded_file:
        st.warning("Feature not enabled.", icon='⚠')
    else:
        st.warning("Please upload PDF", icon='⚠')
