from functools import reduce
import json
import os
import requests
import warnings

import chromadb
import pandas as pd
from chromadb.api.models.Collection import Collection as ChromaCollection
from datasets import load_dataset, Dataset
from getpass import getpass
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.runnables.base import RunnableSequence
from langchain_community.document_loaders import WebBaseLoader, PolarsDataFrameLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from operator import itemgetter
import optuna
import pandas as pd
import plotly.express as px
import polars as pl
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness
)
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional

# Providing api key for OPENAI
#os.environ["OPENAI_API_KEY"] = 

# Getting example docs into vectordb
urls = ["https://en.wikipedia.org/wiki/Hanni_(singer)"]

wikis_loader = WebBaseLoader(urls)
wikis_documents = wikis_loader.load()
#print( wikis_documents[0] )

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

'''
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1250,
    chunk_overlap = 100,
    length_function = len,
    is_separator_regex = False
)
'''

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", deployment="text-embedding-3-small")
#embeddings = OllamaEmbeddings(model="mxbai-embed-large")

'''
splited_docs = text_splitter.split_documents(wikis_documents)
vectorstore = Chroma.from_documents(
    documents=splited_docs, 
    embedding=embeddings, 
    persist_directory="./Vectorstore/chromadb"
)
'''

from langchain_community.chat_models import ChatOllama

'''
llm_options = {
    "Cohere - command-light": lambda: ChatCohere(model_name="command-light", temperature=temperature, max_tokens=max_tokens),
    "Cohere - command": lambda: ChatCohere(model_name="command", temperature=temperature, max_tokens=max_tokens),
    "Cohere - command-r": lambda: ChatCohere(model_name="command-r", temperature=temperature, max_tokens=max_tokens),
    "OpenAI - gpt-3.5-turbo": lambda: ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature, max_tokens=max_tokens),
    "OpenAI - gpt-4-turbo-preview": lambda: ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=temperature, max_tokens=max_tokens),
    "OpenAI - gpt-4": lambda: ChatOpenAI(model_name="gpt-4", temperature=temperature, max_tokens=max_tokens),
    "Anthropic - claude-3-opus-20240229": lambda: ChatAnthropic(model_name="claude-3-opus-20240229", temperature=temperature, max_tokens=max_tokens),
    "Anthropic - claude-3-sonnet-20240229": lambda: ChatAnthropic(model_name="claude-3-sonnet-20240229", temperature=temperature, max_tokens=max_tokens),
    "Anthropic - claude-3-haiku-20240307": lambda: ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=temperature, max_tokens=max_tokens),
    "Ollama - Gemma": lambda: ChatOllama(model_name="gemma", temperature=temperature, max_tokens=1024),
}
'''

llm35 = ChatOpenAI(model="gpt-3.5-turbo")
llm4 = ChatOpenAI(model="gpt-4-turbo")
llm_llama3 = ChatOllama(model="llama3:latest")
generator_llm = llm_llama3
#critic_llm = llm_llama3
#generator_llm = llm35
critic_llm = llm35

example_generator=None
example_generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings,
    chunk_size=1024
)

# distributions: simple, reasoning, multi_context, conditional, conversational
my_distributions = {simple: 0.5, reasoning:0.5}

testset = example_generator.generate_with_langchain_docs(wikis_documents, 2, distributions=my_distributions)
test_df = testset.to_pandas()
# select columes among question,contexts,ground_truth,evolution_type,metadata,and episode_done 
columns_to_show = ['question', 'ground_truth', 'contexts']  # 원하는 열 이름 select
print(test_df[columns_to_show].head())

# save to file
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'question_evolution_results_{timestamp}.csv'
test_df.to_csv(filename, index=False)
print(f"### 결과가 '{filename}' 파일로 저장되었습니다.")


import shutil
from langchain.chains import RetrievalQA

# vectorstore를 저장할 디렉토리 지정
persist_directory = 'db'

# 기존 vectorstore 삭제
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)
    print(f"기존 vectorstore ({persist_directory})를 삭제했습니다.")

# 새로운 vectorstore 생성
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
splits = text_splitter.split_documents(wikis_documents)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_directory)
vectorstore.persist()
print(f"새로운 vectorstore를 생성하고 {persist_directory}에 저장했습니다.")

# 이후 QA 체인 설정 등의 코드...

qa_chain = RetrievalQA.from_chain_type(
    llm=llm_llama3,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# QA 체인을 통한 답변 생성
answers = []
for question in test_df['question']:
    result = qa_chain({"query": question})
    answers.append(result['result'])

# 답변을 테스트셋에 추가
test_df['answer'] = answers

columns_to_show = ['question', 'ground_truth', 'contexts', 'answer']  # 원하는 열 이름 select
print(test_df[columns_to_show].head())

# 현재 시간을 이용한 타임스탬프 생성
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 답변이 추가된 파일명
filename_with_answers = f'question_evolution_results_with_answers_{timestamp}.csv'

# 업데이트된 test_df를 새 파일로 저장
test_df.to_csv(filename_with_answers, index=False)

print(f"답변이 추가된 파일이 '{filename_with_answers}'로 저장되었습니다.")

# 평가 수행 
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall
)
from datasets import Dataset

# test_df를 Hugging Face Dataset 형식으로 변환
dataset = Dataset.from_pandas(test_df)

# 평가 메트릭 정의
metrics = [
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall
]

# Ragas 평가 실행
evaluation_result = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=llm_llama3,  # 이전에 정의한 Ollama LLM 사용
    raise_exceptions=False
)

# 평가 결과 출력
print("\n===== Ragas 평가 결과 =====")
print(evaluation_result)

# 결과를 딕셔너리로 변환
result_dict = {}
for key, value in evaluation_result.items():
    if isinstance(value, float):  # 숫자 값만 포함
        result_dict[key] = value

# DataFrame으로 변환
result_df = pd.DataFrame([result_dict])

# CSV 파일로 저장
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f'ragas_evaluation_results_{timestamp}.csv'
result_df.to_csv(csv_filename, index=False)
print(f"\n평가 결과가 '{csv_filename}' 파일로 저장되었습니다.")

# 세부 평가 점수 출력
print("\n===== 세부 평가 점수 =====")
for metric_name, score in result_dict.items():
    if not pd.isna(score):  # NaN 값 제외
        print(f"{metric_name}: {score:.4f}")
    else:
        print(f"{metric_name}: NaN (평가 실패)")

# NaN 값에 대한 경고 출력
if any(pd.isna(score) for score in result_dict.values()):
    print("\n주의: 일부 메트릭에서 NaN 값이 발생했습니다. 이는 평가 과정에서 문제가 있었음을 나타냅니다.")
