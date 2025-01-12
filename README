# Hydra like agent declaration

This project demonstrates how to orchestrate multiple LangChain agents using a **Hydra-style YAML configuration** (though it does **not** strictly rely on the [Hydra framework](https://github.com/facebookresearch/hydra) itself). Instead, we employ a single YAML file (`hydra_config.yaml`) to define multiple agents, each with its own chain type, model, memory, and more. 

Below is an overview of the core technical details, focusing on how to configure and run these agents in a Hydra-like manner.

---

## Key Technical Components

1. **`hydra_config.yaml`**  
  - Serves as a central configuration file, similar to Hydra’s YAML-based approach.  
  - Contains a top-level `agents` list. Each element in `agents` describes:
    - **Agent name**
    - **LLM provider and model** (e.g., OpenAI GPT-4o-mini)
    - **Chain type** (LLMChain, ConversationalRetrievalChain, etc.)
    - **Memory settings** (optional)
    - **Vector store options** (if needed for retrieval-based chains)
    - **Tools** (for agent tool usage, such as a `CalculatorTool`)

2. **Argparse for Overrides**  
  - We use `argparse` to parse the command-line argument s`--override_llm_model`, which will replace the default model name in all agent configurations.  
  - This mechanism loosely mimics Hydra’s concept of overriding config parameters via the command line, but without strictly using Hydra’s own CLI.

3. **`LangChainAgent` Class**  
  - Responsible for initializing a chain based on the agent’s `chain_type`.  
  - Shows how to wire up memory, vector stores, tools, and prompts in a cohesive manner.

4. **`MultiAgentOrchestrator`**  
  - Reads the list of agents from `hydra_config.yaml`.  
  - Instantiates each `LangChainAgent`.  
  - Runs them all in sequence through `run_all()`.

---

## How to Use

1. **Install Dependencies**

  Make sure you have the required dependencies installed:
  ```bash
  pip install --upgrade langchain openai faiss-cpu pyyaml
  ```
  > If you are using a GPU version of FAISS or other specialized versions, install accordingly.

2. **Set Your OpenAI API Key**  
  ```bash
  export OPENAI_API_KEY="your_openai_api_key"
  ```
  Or load it from a `.env` file or another method of your choice.

3. **Review `hydra_config.yaml`**  
  An example YAML might look like this:
  ```yaml
  agents:
    - name: ExampleAgent
     llm_provider: openai
     llm_model_name: gpt-4o-mini
     chain_type: LLMChain
     prompt: "Hello from LLMChain!"
     memory:
      type: ConversationBufferMemory
      memory_key: chat_history

    - name: RetrievalAgent
     llm_provider: openai
     llm_model_name: gpt-4o-mini
     chain_type: ConversationalRetrievalChain
     vectorstore_type: faiss
     vectorstore_info:
      index_path: "./faiss_index"
     memory:
      type: ConversationBufferMemory
      memory_key: chat_history

    - name: QASourceAgent
     llm_provider: openai
     llm_model_name: gpt-4o-mini
     chain_type: QAWithSourcesChain
     vectorstore_type: faiss
     vectorstore_info:
      index_path: "./faiss_index"

    - name: SummarizeAgent
     llm_provider: openai
     llm_model_name: gpt-4o-mini
     chain_type: SummarizeChain

    - name: ToolAgent
     llm_provider: openai
     llm_model_name: gpt-4o-mini
     chain_type: ToolUsingAgent
     tools:
      - type: CalculatorTool
  ```
  This file follows a Hydra-like structure: a single file containing multiple agent configurations.

4. **Run the Main Script**  
  Simply execute:
  ```bash
  python main.py
  ```
  The script does the following:
  - Reads `hydra_config.yaml`.
  - Initializes each agent with the specified chain and optional memory, tool, or vector store.  
  - Invokes `run_all()` to demonstrate each chain’s functionality.

5. **Override the LLM Model** *(Hydra-Style Override)*  
  While we are not using Hydra’s CLI directly, you can still emulate Hydra’s override style:
  ```bash
  python main.py --override_llm_model gpt-4o-mini
  ```
  This command updates **every** agent’s `llm_model_name` to `gpt-4o-mini`, providing a single point of control from the command line—akin to Hydra’s override mechanism.

---

## Technical Flow

1. **Loading Configuration**  
  - `main.py` uses `argparse` to parse command-line arguments (e.g., `--override_llm_model`).
  - Loads `hydra_config.yaml` via `pyyaml` (`yaml.safe_load`).
  - If an override is present, updates all `llm_model_name` fields.

2. **Agent Initialization**  
  - For each agent config, `LangChainAgent` sets up:
    - **LLM** (via `ChatOpenAI` with the chosen model)
    - **Memory** (if specified)
    - **Chain Type** (LLMChain, ConversationalRetrievalChain, etc.)
    - **Vector Store** (FAISS, if needed)
    - **Tools** (e.g., a `CalculatorTool`)

3. **Running the Orchestrator**  
  - `MultiAgentOrchestrator.run_all()` loops through every agent, invoking `agent.run()`.
  - Each agent’s `run()` method delegates to the underlying chain’s `run()` or `__call__` method.

4. **Console Output**  
  - Results are printed in the console, showing chain outputs (LLM text, retrieval info, summarized content, etc.).

---

## Adapting to True Hydra Workflow

- If you decide to use [Hydra](https://github.com/facebookresearch/hydra) in a more official capacity, you would typically:
  - Store configurations in multiple `.yaml` files (e.g., `conf/agent/agent1.yaml`, `conf/agent/agent2.yaml`, etc.).
  - Use Hydra’s command-line interface to select or override these configs dynamically.
  - Integrate Hydra’s decorators (e.g., `@hydra.main()`) in `main.py`.
- The current example is intentionally minimal, using a single `hydra_config.yaml` and `argparse` for command-line parameters. Converting this to a **full Hydra** project would mostly involve reorganizing the config files and adjusting the entry-point function.

---

## Additional Notes

- **Memory Management**: We demonstrate `ConversationBufferMemory` for multi-turn conversations. You could swap in other memory classes (e.g., `VectorStoreRetrieverMemory`) as needed.
- **Prompt Engineering**: `PromptTemplate` usage for LLMChain. If you require more advanced prompts, integrate [ChatPromptTemplate](https://python.langchain.com/docs/modules/model_io/prompts/chat_prompt_template) or custom logic.
- **Vector Stores**: The example uses local FAISS loading. You can also integrate Pinecone, Chroma, Weaviate, or other stores with minimal changes in `_load_vectorstore()`.
- **ToolUsage**: Tools are added to an agent’s toolset, enabling the chain to delegate certain queries to specialized routines like `CalculatorTool`.

---

## Conclusion

This setup showcases a **Hydra-style** approach to configuring multiple LangChain agents. You get a central YAML config for all agent definitions, straightforward command-line overrides, and an extensible architecture for new chain types or tools.

Feel free to modify the code to suit your exact needs, or consider adopting Hydra itself for even more flexible configurations.