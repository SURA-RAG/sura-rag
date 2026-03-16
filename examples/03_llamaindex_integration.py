# Run this example: python examples/03_llamaindex_integration.py
# Prerequisites: pip install sura-rag[llamaindex]
#                ollama serve && ollama pull llama3.2:3b

"""
LlamaIndex Integration Example

Demonstrates using SuraQueryEngineWrapper as a drop-in wrapper
for any LlamaIndex query engine, automatically scanning responses
through the SURA guardrail.
"""

import sura_rag as sr


def main():
    print("=== SURA-RAG LlamaIndex Integration ===\n")

    try:
        from sura_rag.connectors.llamaindex import SuraQueryEngineWrapper
    except ImportError:
        print(
            "LlamaIndex is not installed. Install with:\n"
            "  pip install sura-rag[llamaindex]"
        )
        return

    # In a real app:
    #
    # from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
    #
    # # Load and index documents
    # documents = SimpleDirectoryReader("data").load_data()
    # index = VectorStoreIndex.from_documents(documents)
    #
    # # Create SURA client
    # adapter = sr.adapters.ChromaDBAdapter("my_collection")
    # client = sr.SuraClient(vector_store=adapter)
    #
    # # Wrap the query engine
    # base_engine = index.as_query_engine()
    # safe_engine = SuraQueryEngineWrapper(
    #     base_engine=base_engine,
    #     sura_client=client,
    # )
    #
    # # Query as usual — responses are automatically guardrailed
    # response = safe_engine.query("What is John's salary?")
    # print(response)
    #
    # # Inspect the scan result
    # if safe_engine.last_scan:
    #     print(f"Leaked: {safe_engine.last_scan.leaked}")
    #     print(f"Action: {safe_engine.last_scan.action_taken}")

    print("LlamaIndex integration ready.")
    print("See the code comments for usage patterns.")


if __name__ == "__main__":
    main()
