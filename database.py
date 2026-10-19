from contextlib import contextmanager
from sqlalchemy import create_engine
from langchain_postgres import PGVector
from config import get_database_url, get_collection_name, get_embedding_function


# Creates access to PGVector when open_database is used
@contextmanager
def open_database(): # Machine Generated
    engine = create_engine(get_database_url(), connect_args={"connect_timeout": 5})
    try:
        yield PGVector(
            connection=engine,
            embeddings=get_embedding_function(),
            collection_name=get_collection_name(),
            use_jsonb=True,
        )
    finally:
        engine.dispose()
