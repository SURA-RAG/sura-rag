# Run this example: python examples/02_langchain_integration.py
# Prerequisites: pip install sura-rag[langchain]
#                ollama serve && ollama pull llama3.2:3b

"""
LangChain Integration Example

Demonstrates using SuraRetriever as a drop-in replacement
for any LangChain retriever, automatically filtering out
forgotten documents at the retrieval layer.
"""

import sura_rag as sr


def main():
    print("=== SURA-RAG LangChain Integration ===\n")

    # This example requires LangChain to be installed
    try:
        from langchain_community.vectorstores import Chroma
        from sura_rag.connectors.langchain import SuraRetriever
    except ImportError:
        print(
            "LangChain is not installed. Install with:\n"
            "  pip install sura-rag[langchain]"
        )
        return

    # Set up ChromaDB with LangChain
    # In a real app, you'd use your existing LangChain setup
    print("Setting up LangChain with ChromaDB...")

    # Create SURA client
    adapter = sr.adapters.ChromaDBAdapter(
        collection_name="langchain_demo",
        persist_directory="./langchain_demo_db",
    )
    client = sr.SuraClient(
        vector_store=adapter,
        config=sr.SuraConfig(enable_rich_logging=False),
    )

    # Wrap existing retriever
    # base_retriever = vectorstore.as_retriever()
    # sura_retriever = SuraRetriever(
    #     base_retriever=base_retriever,
    #     sura_client=client,
    #     filter_mode="hard_block",
    # )
    #
    # # Use in any LangChain chain — unchanged
    # chain = RetrievalQA.from_chain_type(
    #     llm=ChatOllama(model="llama3.2:3b"),
    #     retriever=sura_retriever,
    # )
    # result = chain.invoke("What is John's salary?")

    print("LangChain integration ready.")
    print("See the code comments for usage patterns.")


if __name__ == "__main__":
    main()
