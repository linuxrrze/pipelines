"""
title: Llama Index Ollama Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings.
requirements: llama-index, llama-index-vector-stores-chroma, llama-index-llms-ollama, llama-index-embeddings-ollama, chromadb
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import os

from pydantic import BaseModel


class Pipeline:

    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str
        LLAMAINDEX_DB_DIR: str
        LLAMAINDEX_INPUT_DIR: str

    def __init__(self):
        self.documents = None
        self.index = None
        self.name = "RAG ChromaDB"

        self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text"),
                "LLAMAINDEX_DB_DIR": os.getenv("LLAMAINDEX_DB_DIR", "/app/backend/data/db/"),
                "LLAMAINDEX_INPUT_DIR": os.getenv("LLAMAINDEX_INPUT_DIR", "/app/backend/data/input"),
            }
        )

    async def on_startup(self):
        import chromadb
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
        from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.core import StorageContext
	
        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )

        # This function is called when the server is started.
        global documents, index

        self.documents = SimpleDirectoryReader(self.valves.LLAMAINDEX_INPUT_DIR, recursive=True).load_data()

        # initialize client, setting path to save data
        db = chromadb.PersistentClient(path=self.valves.LLAMAINDEX_DB_DIR+"/chroma_db")

        # create collection
        chroma_collection = db.get_or_create_collection("quickstart")

        # assign chroma as the vector_store to the context
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self.index = VectorStoreIndex.from_documents(self.documents, storage_context=storage_context)

        for doc in self.documents:
            print(f"Indexing {doc}...")
            self.index.insert(doc)

        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        print(messages)
        print(user_message)

        query_engine = self.index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)

        return response.response_gen
