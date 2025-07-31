from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel,Field
from typing import Literal

load_dotenv()

model = ChatOpenAI()
parser1 = StrOutputParser()

class FeedSent(BaseModel):
    sentiment : Literal['positive','negative'] = Field(description='sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=FeedSent)

prompt1 = PromptTemplate(
    template='Give me the sentiment of the following feedback.\n{feedback}\n{format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="Give me appropriate message to this positive feedback.\n{feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="Give me appropriate message to this negative feedback.\n{feedback}",
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser1),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser1),
    RunnableLambda(lambda x:"could not find sentiment")
)

chain = classifier_chain | branch_chain

chain.invoke('this is a terrible car')

chain.get_graph().print_ascii()