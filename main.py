import time
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
# prompts
from langchain import PromptTemplate, LLMChain
import textwrap

class CFG:
    # LLMs
    model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    temperature = 0.5
    top_p = 0.95
    repetition_penalty = 1.15
    do_sample = True
    max_new_tokens = 1024
    num_return_sequences=1

    # splitting
    split_chunk_size = 800
    split_overlap = 0
    
    # embeddings
    embeddings_model_repo = 'sentence-transformers/all-MiniLM-L6-v2'

    # similar passages
    k = 5
    
    # paths
    PDFs_path = '/Users/binhle/project/llm/model/hp_books/'
    # PDFs_path = '/kaggle/input/data-visualization-cheat-cheats-and-resources/Data visualizaion Books'
    Embeddings_path =  '/Users/binhle/project/llm/model/faiss_hp/'
    Persist_directory = '/Users/binhle/project/llm/model/faiss_hp/harry-potter-vectordb'
    # Persist_directory = './data-viz-vectordb'

llm = HuggingFaceHub(
    repo_id = CFG.model_name,
    model_kwargs={
        "max_new_tokens": CFG.max_new_tokens,
        "temperature": CFG.temperature,
        "top_p": CFG.top_p,
        "repetition_penalty": CFG.repetition_penalty,
        "do_sample": CFG.do_sample,
        "num_return_sequences": CFG.num_return_sequences        
    },
    huggingfacehub_api_token = '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'
)



from langchain.embeddings import HuggingFaceInstructEmbeddings
### download embeddings model
embeddings = HuggingFaceInstructEmbeddings(
    model_name = CFG.embeddings_model_repo,
    model_kwargs = {"device": "cpu"}
)

### load vector DB embeddings
vectordb = FAISS.load_local(
    CFG.Embeddings_path,
    embeddings
)

retriever = vectordb.as_retriever(search_kwargs = {"k": CFG.k, "search_type" : "similarity"})

# we get the context part by embedding retrieval 
prompt_template = """<s>[INST] You are given the context after <<CONTEXT>> and a question after <<QUESTION>>.

Answer the question by only using the information in context. Only base your answer on the information in the context. Even if you know something more,
keep silent about it. It is important that you only tell what can be infered from the context alone.

<<QUESTION>>{question}\n<<CONTEXT>>{context} [/INST]"""

PROMPT = PromptTemplate(
    template = prompt_template, 
    input_variables = ["question", "context"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff", # map_reduce, map_rerank, stuff, refine
    retriever = retriever, 
    chain_type_kwargs = {"prompt": PROMPT},
    return_source_documents = True,
    verbose = False
)

def wrap_text_preserve_newlines(text, width=700):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def process_llm_response(llm_response):
    ans = wrap_text_preserve_newlines(llm_response['result'])
    
    sources_used = ' \n \n > '.join(
        [
            source.metadata['source'].split('/')[-1][:-4] + ' - page: ' + str(source.metadata['page'])
            for source in llm_response['source_documents']
        ]
    )
    
    ans = ans + ' \n \n \t Sources: \n >' + sources_used
    return ans

def llm_ans(query):
    start = time.time()
    llm_response = qa_chain(query)
    ans = process_llm_response(llm_response)
    end = time.time()

    time_elapsed = int(round(end - start, 0))
    time_elapsed_str = f' \n \n Time elapsed: {time_elapsed} s'
    return ans.strip() + time_elapsed_str


question = st.text_input("Asking question:")
submit = st.button('Generate Answer!')

st.write("---")

if submit or question:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Answer from LLM model (Mistral-7B-Instruct-v0.1):")
        with st.spinner(text="This may take a moment..."):
            answer_llm = (llm(f"""<s>[INST] {question} [/INST]""", raw_response=True).strip())
        st.markdown(answer_llm)
    with col2:
        st.subheader("Answer from Fine-tunning LLM model:")
        with st.spinner(text="This may take a moment..."):
            answer_tune= llm_ans(question)
        st.markdown(answer_tune)