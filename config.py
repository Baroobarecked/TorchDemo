import hashlib
import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy.engine import make_url
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data"
load_dotenv(ROOT / ".env")

# There are two supported providers, Ollama and OpenAI, this loads whichever is set in .env
def get_provider(): # Machine Generated
    provider = os.getenv("PROVIDER", "ollama").strip().lower()
    if provider not in {"openai", "ollama"}:
        raise ValueError("PROVIDER must be openai or ollama")
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY to use OpenAI")
    return provider

# Gets the name of the correct embedding model per provider
def get_embedding_model(): # Machine Generated
    if get_provider() == "openai":
        return os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    return os.getenv("OLLAMA_EMBED_MODEL")

# Sets the langchain embedding model
def get_embedding_function():
    if get_provider() == "openai":
        return OpenAIEmbeddings(model=get_embedding_model(), request_timeout=90)
    return OllamaEmbeddings(
        model=get_embedding_model(),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        client_kwargs={"timeout": 300},
    )

# Each provider has it's own embedding collection in PGVector, this gets that ID
def get_collection_name(): # Machine Generated
    identity = f"{get_provider()}:{get_embedding_model()}"
    if get_provider() == "ollama":
        identity += ":" + os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return "board_games_" + hashlib.sha256(identity.encode()).hexdigest()[:16]

# Gets the database url from .env
def get_database_url(): # Machine Generated
    value = os.getenv("APP_DATABASE_URL")
    if not value:
        raise ValueError("Set APP_DATABASE_URL in .env")
    return make_url(value).set(drivername="postgresql+psycopg")

# Sets up langchain's chat to use the provider's model
def get_chat_model(): # Machine Generated
    if get_provider() == "openai":
        return ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-6-luna"), timeout=90)
    return ChatOllama(
        model=os.getenv("OLLAMA_CHAT_MODEL"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
        client_kwargs={"timeout": 300}
    )