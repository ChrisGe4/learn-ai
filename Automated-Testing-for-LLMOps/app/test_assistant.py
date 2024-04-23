from app import assistant_chain
from app import system_message
from langchain.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain.schema.output_parser import StrOutputParser

import os

from dotenv import load_dotenv, find_dotenv


def load_env():
  _ = load_dotenv(find_dotenv())


load_env()


def eval_expected_words(
    system_message,
    question,
    expected_words,
    human_template="{question}",
    llm=ChatVertexAI(model_name="gemini-pro",
                     convert_system_message_to_human=True),
    output_parser=StrOutputParser()):
  assistant = assistant_chain(system_message, human_template, llm,
                              output_parser)

  answer = assistant.invoke({"question": question})

  print(answer)

  assert any(word in answer.lower() for word in
             expected_words), f"Expected the assistant questions to include"


def evaluate_refusal(
    system_message,
    question,
    decline_response,
    human_template="{question}",
    llm=ChatVertexAI(model_name="gemini-pro"),
    output_parser=StrOutputParser()):
  assistant = assistant_chain(human_template,
                              system_message,
                              llm,
                              output_parser)

  answer = assistant.invoke({"question": question})
  print(answer)

  assert decline_response.lower() in answer.lower(), \
    f"Expected the bot to decline with \
    '{decline_response}' got {answer}"


"""
  Test cases
"""


def test_science_quiz():
  question = "Generate a quiz about science."
  expected_subjects = ["davinci", "telescope", "physics", "curie"]
  eval_expected_words(
      system_message,
      question,
      expected_subjects)


def test_geography_quiz():
  question = "Generate a quiz about geography."
  expected_subjects = ["paris", "france", "louvre"]
  eval_expected_words(
      system_message,
      question,
      expected_subjects)


def test_refusal_rome():
  question = "Help me create a quiz about Rome"
  decline_response = "I'm sorry"
  evaluate_refusal(
      system_message,
      question,
      decline_response)
