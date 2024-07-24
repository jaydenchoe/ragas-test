import os
import shutil
from datetime import datetime

import pandas as pd
import streamlit as st
from datasets import Dataset
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader, PolarsDataFrameLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_relevancy,
    answer_correctness,
)
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning

# Load environment variables
load_dotenv()

st.title("QA Set Generator and Evaluator")

# Main selection
option = st.selectbox("Select an option", ["Generate QA Set", "Evaluate Existing QA Set"])

# Function to generate QA set and run QA chain
def generate_and_run_qa_chain(documents=None, use_url=True, from_existing_file=False):
    if from_existing_file:
        # File upload for existing data
        uploaded_file = st.file_uploader("Upload file with question, original context, and ground truth", type=["csv"])
        if uploaded_file is not None:
            test_df = pd.read_csv(uploaded_file)
            st.write("File loaded successfully.")
    else:
        st.header("Generating QA Set")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", deployment="text-embedding-3-small")
        llm4omini = ChatOpenAI(model="gpt-4o-mini")
        llm_llama31 = ChatOllama(model="llama3.1:latest")
        generator_llm = llm_llama31
        critic_llm = llm4omini

        example_generator = TestsetGenerator.from_langchain(
            generator_llm,
            critic_llm,
            embeddings,
            chunk_size=1024
        )

        my_distributions = {simple: 0.5, reasoning: 0.5}
        testset = example_generator.generate_with_langchain_docs(documents, 2, distributions=my_distributions)
        test_df = testset.to_pandas()
        test_df.rename(columns={'contexts': 'original_contexts'}, inplace=True)

    st.header("Running QA Chain")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
    # Ensure all items in 'original_contexts' are strings
    doc_objects = [Document(page_content=str(text)) for text in test_df['original_contexts']]
    splits = text_splitter.split_documents(documents=doc_objects)
    
    # Check if FAISS DB directory exists, if so, delete it
    faiss_db_dir = "faiss_db"
    if os.path.exists(faiss_db_dir):
        shutil.rmtree(faiss_db_dir)
    
    vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings(model="text-embedding-3-small"), faiss_db_dir)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    answers = []
    retrieved_contexts = []
    for question in test_df['question']:
        result = qa_chain({"query": question})
        answers.append(result['result'])
        sources = result["source_documents"]
        contents = []
        for i in range(len(sources)):
            contents.append(sources[i].page_content)
        retrieved_contexts.append(contents)

    test_df['answer'] = answers
    test_df['contexts'] = retrieved_contexts

    columns_to_show = ['question', 'ground_truth', 'original_contexts', 'contexts', 'answer']
    st.write("Generated answers:")
    st.write(test_df[columns_to_show].head())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_with_answers = f'question_evolution_results_with_answers_{timestamp}.csv'
    test_df.to_csv(filename_with_answers, index=False)
    st.write(f"Results with answers saved to `{filename_with_answers}`")

    return test_df, filename_with_answers

# Function to evaluate QA set
def evaluate_qa_set(test_df):
    st.header("Evaluating QA Set")
    
    # Ensure 'contexts' column is of type Sequence[string]
    if isinstance(test_df['contexts'].iloc[0], str):
        test_df['contexts'] = test_df['contexts'].apply(lambda x: [x])
    elif isinstance(test_df['contexts'].iloc[0], list):
        test_df['contexts'] = test_df['contexts'].apply(lambda x: [str(item) for item in x])
    else:
        raise ValueError("Unsupported data format in 'contexts' column")
    
    dataset = Dataset.from_pandas(test_df)
    metrics = [faithfulness, context_relevancy, answer_correctness]

    evaluation_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ChatOpenAI(model="gpt-4o-mini"),
        raise_exceptions=False
    )

    result_df = evaluation_result.to_pandas()
    st.write("Evaluation results:")
    st.write(result_df.head())

    csv_filename = f'ragas_evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    result_df.to_csv(csv_filename, index=False)
    st.write(f"Evaluation results saved to `{csv_filename}`")

# Manage session state for buttons
if 'generate_button' not in st.session_state:
    st.session_state.generate_button = False

if 'evaluate_button' not in st.session_state:
    st.session_state.evaluate_button = False

# Generate QA Set
if option == "Generate QA Set":
    generation_option = st.selectbox("Select generation option", ["Create from scratch", "Use existing file"])
    
    if generation_option == "Create from scratch":
        source_option = st.selectbox("Select source type", ["PDF File", "URL"])
        
        if source_option == "URL":
            url = st.text_input("Enter the URL", value="https://en.wikipedia.org/wiki/Hanni_(singer)")
            if st.button("Load and Generate QA Set"):
                wikis_loader = WebBaseLoader([url])
                wikis_documents = wikis_loader.load()
                test_df, saved_filename = generate_and_run_qa_chain(wikis_documents, use_url=True, from_existing_file=False)
                st.session_state.generate_button = True
                st.session_state.test_df = test_df
                st.session_state.saved_filename = saved_filename
        
        elif source_option == "PDF File":
            pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
            if st.button("Load and Generate QA Set") and pdf_file is not None:
                pdf_documents = PolarsDataFrameLoader(pdf_file).load()
                test_df, saved_filename = generate_and_run_qa_chain(pdf_documents, use_url=False, from_existing_file=False)
                st.session_state.generate_button = True
                st.session_state.test_df = test_df
                st.session_state.saved_filename = saved_filename

    elif generation_option == "Use existing file":
        test_df, saved_filename = generate_and_run_qa_chain(from_existing_file=True)
        st.session_state.generate_button = True
        st.session_state.test_df = test_df
        st.session_state.saved_filename = saved_filename

    if st.session_state.generate_button:
        if st.button("Evaluate this QA Set"):
            evaluate_qa_set(st.session_state.test_df)
            st.session_state.evaluate_button = True

# Evaluate Existing QA Set
elif option == "Evaluate Existing QA Set":
    csv_file = st.file_uploader("Upload CSV", type=["csv"])
    evaluate_button = st.button("Evaluate QA Set")
    if evaluate_button and csv_file is not None:
        test_df = pd.read_csv(csv_file)
        # Ensure 'contexts' column is of type Sequence[string]
        if isinstance(test_df['contexts'].iloc[0], str):
            test_df['contexts'] = test_df['contexts'].apply(lambda x: [x])
        elif isinstance(test_df['contexts'].iloc[0], list):
            test_df['contexts'] = test_df['contexts'].apply(lambda x: [str(item) for item in x])
        else:
            raise ValueError("Unsupported data format in 'contexts' column")
        evaluate_qa_set(test_df)
