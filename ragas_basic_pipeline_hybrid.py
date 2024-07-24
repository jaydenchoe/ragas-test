import json
import os
import requests
import warnings
from datetime import datetime

import pandas as pd
from datasets import load_dataset, Dataset
from getpass import getpass

import chromadb
from chromadb.api.models.Collection import Collection as ChromaCollection


from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.runnables.base import RunnableSequence
from langchain_community.document_loaders import WebBaseLoader, PolarsDataFrameLoader
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy, # important
    faithfulness, # important
    context_recall,
    context_precision,
    context_relevancy,
    answer_correctness, # most important
)
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional

import shutil

from functools import reduce
from operator import itemgetter
import optuna
import pandas as pd
import plotly.express as px
import polars as pl

from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# 스크립트 파일의 디렉토리로 작업 디렉토리 변경
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# .env 파일에서 환경 변수 로드
# Providing api keys for OPENAI and ANTHROPIC
load_dotenv()

# Getting example docs into vectordb
urls = ["https://en.wikipedia.org/wiki/Hanni_(singer)"]

wikis_loader = WebBaseLoader(urls)
wikis_documents = wikis_loader.load()
#print( wikis_documents[0] )

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", deployment="text-embedding-3-small")
#embeddings = OllamaEmbeddings(model="mxbai-embed-large") # smallest but so slow!!!!

llm4omini = ChatOpenAI(model="gpt-4o-mini")
#llm4 = ChatOpenAI(model="gpt-4-turbo")
#llm_llama3 = ChatOllama(model="llama3:latest")
llm_llama31 = ChatOllama(model="llama3.1:latest")
#llm_haiku3 = ChatAnthropic(model="claude-3-haiku-20240307")

generator_llm = llm_llama31
critic_llm = llm4omini

example_generator=None
example_generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings,
    chunk_size=1024
)

# distributions: simple, reasoning, multi_context, conditional, conversational
my_distributions = {simple: 0.5, reasoning:0.5}

## 테스트셋 생성 
testset = example_generator.generate_with_langchain_docs(wikis_documents, 2, distributions=my_distributions)
test_df = testset.to_pandas()

# Rename 'contexts' column to 'original_context' for RAGAS evaluation
test_df.rename(columns={'contexts': 'original_contexts'}, inplace=True)

# select columes among question,contexts,ground_truth,evolution_type,metadata,and episode_done 
columns_to_show = ['question', 'ground_truth', 'original_contexts']  # 원하는 열 이름 select
print(test_df[columns_to_show].head())

# 중간 결과를 CSV 파일로 저장
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'question_evolution_results_{timestamp}.csv'
test_df.to_csv(filename, index=False)
print(f"### 결과가 '{filename}' 파일로 저장되었습니다.")


### 여기서부터는 QA Chain 구현과 실행

# vectorstore를 저장할 디렉토리 지정
persist_directory = 'db'

# 기존 vectorstore 삭제 후 신규 vectorstore 생성해서 문서 저장
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)
    print(f"기존 vectorstore ({persist_directory})를 삭제했습니다.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
splits = text_splitter.split_documents(wikis_documents)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_directory)
vectorstore.persist()
print(f"새로운 vectorstore를 생성하고 {persist_directory}에 저장했습니다.")

# QA 체인 설정
qa_chain = RetrievalQA.from_chain_type(
    llm=llm4omini,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# QA 체인을 통한 답변 생성
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

# 답변과 새로운 contexts를 테스트셋에 추가
test_df['answer'] = answers
test_df['contexts'] = retrieved_contexts

columns_to_show = ['question', 'ground_truth', 'original_contexts', 'contexts', 'answer']  # 원하는 열 이름 select
print(test_df[columns_to_show].head())

# 답변이 추가된 테스트셋을 파일로 저장
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename_with_answers = f'question_evolution_results_with_answers_{timestamp}.csv'
test_df.to_csv(filename_with_answers, index=False)
print(f"답변이 추가된 파일이 '{filename_with_answers}'로 저장되었습니다.")

### 평가 준비와 실행
dataset = Dataset.from_pandas(test_df)

# 평가 메트릭 정의
metrics = [
    faithfulness,
#    answer_relevancy,
    context_relevancy,
#    context_recall,
    answer_correctness
]

# Ragas 평가 실행
evaluation_result = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=llm4omini,  
    raise_exceptions=False
)

# 평가 결과를 DataFrame으로 변환
result_df = evaluation_result.to_pandas()
print( result_df.head())

# 평가 결과를 CSV 파일로 저장
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f'ragas_evaluation_results_{timestamp}.csv'
result_df.to_csv(csv_filename, index=False)
print(f"\n평가 결과가 '{csv_filename}' 파일로 저장되었습니다.")
