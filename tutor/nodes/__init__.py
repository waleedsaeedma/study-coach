from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

# The tutor's main brain, shared by all node files: from tutor.nodes import llm
llm = ChatOpenAI(model="gpt-5.4-mini")
