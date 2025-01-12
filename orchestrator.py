import os
import yaml
import logging
import argparse
from typing import Any, Dict, List, Optional

from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.agents.tools import Tool  
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from langchain.chains import (
    LLMChain,
    ConversationalRetrievalChain,

)

from langchain.chains.summarize import load_summarize_chain

from langchain.chains import RetrievalQAWithSourcesChain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "A tool that calculates the given expression and returns the result"

    def _run(self, query: str) -> str:
        try:
            result = eval(query)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    async def _arun(self, query: str) -> str:
        return self._run(query)

class LangChainAgent:
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.chain = None
        self.memory = None
        self._initialize()

    def _initialize(self):
        logger.info(f"Initializing Agent [{self.name}] with config: {self.config}")
        llm_provider = self.config.get("llm_provider", "openai")
        llm_model_name = self.config.get("llm_model_name", "gpt-4o-mini")

        if llm_provider == "openai":
            openai_api_key = os.environ.get("OPENAI_API_KEY", "")

            llm = ChatOpenAI(
                model_name=llm_model_name,
                temperature=0.7,
                openai_api_key=openai_api_key
            )
        else:
            raise NotImplementedError(f"Unsupported LLM provider: {llm_provider}")

        memory_config = self.config.get("memory")
        if memory_config and memory_config.get("type") == "ConversationBufferMemory":
            self.memory = ConversationBufferMemory(
                memory_key=memory_config.get("memory_key", "chat_history"),
                return_messages=True
            )

        chain_type = self.config.get("chain_type", "LLMChain")

        if chain_type == "LLMChain":
            prompt_text = self.config.get("prompt", "Hello from LLMChain!")

            from langchain.prompts import PromptTemplate
            prompt_template = PromptTemplate(
                input_variables=[],
                template=prompt_text
            )
            self.chain = LLMChain(llm=llm, prompt=prompt_template, memory=self.memory)

        elif chain_type == "ConversationalRetrievalChain":
            self.chain = self._init_conversational_retrieval_chain(llm)

        elif chain_type == "QAWithSourcesChain":
            self.chain = self._init_qa_with_sources_chain(llm)

        elif chain_type == "SummarizeChain":
            self.chain = self._init_summarize_chain(llm)

        elif chain_type == "ToolUsingAgent":
            self.chain = self._init_tool_using_agent(llm)

        else:
            raise NotImplementedError(f"Unsupported chain_type: {chain_type}")

    def _init_conversational_retrieval_chain(self, llm):
        vectorstore = self._load_vectorstore()
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=False,
            memory=self.memory
        )
        return chain

    def _init_qa_with_sources_chain(self, llm):
        vectorstore = self._load_vectorstore()

        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
        )
        return chain

    def _init_summarize_chain(self, llm):

        chain = load_summarize_chain(llm, chain_type="map_reduce")
        return chain

    def _init_tool_using_agent(self, llm):
        tool_configs = self.config.get("tools", [])
        tools = []
        for t in tool_configs:
            if t["type"] == "CalculatorTool":
                tools.append(CalculatorTool())
            else:
                raise NotImplementedError(f"Unknown tool type: {t['type']}")

        agent_chain = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        return agent_chain

    def _load_vectorstore(self):
        vectorstore_type = self.config.get("vectorstore_type", "faiss")
        vectorstore_info = self.config.get("vectorstore_info", {})
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY", ""))

        if vectorstore_type == "faiss":
            index_path = vectorstore_info.get("index_path", "./faiss_index")
            vectorstore = FAISS.load_local(index_path, embeddings)
        else:
            raise NotImplementedError(f"Unsupported vectorstore type: {vectorstore_type}")

        return vectorstore

    def run(self, query: Optional[str] = None, docs: Optional[List[str]] = None) -> Any:
        logger.info(f"Running Agent [{self.name}] with query='{query}' docs='{docs}'")

        chain_type = self.config.get("chain_type")

        if chain_type == "LLMChain":
            result = self.chain.run({})
            print(f"[{self.name}] LLMChain Output: {result}")
            return result

        elif chain_type == "ConversationalRetrievalChain":
            user_query = query or "Hi, I'd like to ask something about the docs"
            result = self.chain({"question": user_query})
            print(f"[{self.name}] ConversationalRetrievalChain Output: {result['answer']}")
            return result

        elif chain_type == "QAWithSourcesChain":
            user_query = query or "What's the content of the dummy doc?"
            result = self.chain({"question": user_query})
            print(f"[{self.name}] RetrievalQAWithSourcesChain Answer: {result['answer']}")
            print(f"[{self.name}] Sources: {result['sources']}")
            return result

        elif chain_type == "SummarizeChain":
            if not docs:
                docs = ["No document to summarize."]
            doc_objs = [Document(page_content=text) for text in docs]
            result = self.chain.run({"input_documents": doc_objs})
            print(f"[{self.name}] SummarizeChain Output: {result}")
            return result

        elif chain_type == "ToolUsingAgent":
            user_query = query or "What is 12345 * 678?"
            result = self.chain.run(user_query)
            print(f"[{self.name}] ToolUsingAgent Output: {result}")
            return result

        else:
            logger.warning(f"[{self.name}] Unknown chain type. Skipping run.")
            return None

class MultiAgentOrchestrator:
    def __init__(self, agent_configs: List[Dict[str, Any]]):
        self.agents: List[LangChainAgent] = []
        for agent_config in agent_configs:
            name = agent_config.get('name', 'UnnamedAgent')
            agent = LangChainAgent(name=name, config=agent_config)
            self.agents.append(agent)

    def run_all(self):
        for agent in self.agents:
            chain_type = agent.config.get("chain_type")

            if chain_type in ["LLMChain"]:
                agent.run()

            elif chain_type in ["ConversationalRetrievalChain", "QAWithSourcesChain"]:
                agent.run(query="Can you tell me something about the dummy doc?")

            elif chain_type == "SummarizeChain":
                docs = [
                    "LangChain is a framework for developing applications powered by LLMs. "
                    "It enables chaining together multiple components to create more complex AI applications."
                ]
                agent.run(docs=docs)

            elif chain_type == "ToolUsingAgent":
                agent.run(query="What is 42 * 7?")

            else:
                logger.warning(f"Skipping agent [{agent.name}] - unsupported chain type.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--override_llm_model", type=str, help="Override all agents' llm_model_name")
    args = parser.parse_args()

    with open('hydra_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.override_llm_model:
        logger.info(f"Overriding LLM model name to: {args.override_llm_model}")
        for agent_conf in config.get('agents', []):
            agent_conf['llm_model_name'] = args.override_llm_model

    orchestrator = MultiAgentOrchestrator(config['agents'])
    orchestrator.run_all()

if __name__ == "__main__":
    main()