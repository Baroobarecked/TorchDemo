import argparse
import json
from langchain_core.prompts import ChatPromptTemplate
from database import open_database
from config import get_chat_model
from flow_populate_database import ingest_rulebook

# Machine Generated, Human Edited
PROMPT_TEMPLATE = """
Answer the question based only on the following context.
If the context does not contain the answer, say you do not know and mention the game title if it was given in the question, otherwise say there was no game mentioned.
If there is a game to mention when the question supplies a title and the answer is not found, use this format at the end of the response:
Game title to research: "Game Title"

{context}

---

Answer the question based on the above context: {question}
"""

# Human Genrated
VALIDATE_KNOWN_TEMPLATE = """
Based on the below, was the answer found?
{answer}

---

Return data only in this string format:
{{
    "Valid": true,
    "Game": "Game title"
}}

Set "Valid" to true or false depending on whether
the answer was found or if the game title is found in the context here: 
{sources}

Look for this phrasing- Game title to research: "Game Title", use the "Game Title" in the response.
"""

# Human Genrated
EXTRACT_TITLE_TEMPLATE = """
Extract any game titles from this query:
{query}

---

Return data only in this string format:
{{
    "Game": "Game title"
}}
"""

MAX_DEPTH = 2

def query_rag(query_text: str, current_depth: int = 0): # Human Generated
    current_depth += 1

    # Ensure the query is not empty
    if not query_text.strip():
        raise ValueError("Enter a nonempty question")

    # Get the context via similarity
    with open_database() as db:
        results = db.similarity_search_with_score(query_text, k=5)

    model = get_chat_model()

    # If no content was found, extract the game title, look for a pdf to download, and try again
    if not results:
        extract_prompt = ChatPromptTemplate.from_template(EXTRACT_TITLE_TEMPLATE).format(
            query=query_text
        )

        extracted_string = model.invoke(extract_prompt).content
        try:
            extracted = json.loads(extracted_string)
        except:
            raise ValueError("Error in formatted response: " + extracted_string)
        

        ingest_rulebook(extracted["Game"])

        if(current_depth < MAX_DEPTH):
            return query_rag(query_text, current_depth)
        return "I have failed. Forgive me."

    # Load the context and prompt
    context_text = "\n\n---\n\n".join(doc.page_content for doc, _score in results)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text, 
        question=query_text
    )
    
    # Invoke prompt, collect response
    response_text = model.invoke(prompt).content
    sources = [doc.metadata["id"] for doc, _score in results]

    # Build validation prompt
    validate_response_prompt = ChatPromptTemplate.from_template(VALIDATE_KNOWN_TEMPLATE).format(
        answer=response_text,
        sources=sources
    )

    # Invoke prompt, collect response
    validated_response_string = model.invoke(validate_response_prompt).content
    try:
        validated_response = json.loads(validated_response_string)
    except:
        raise ValueError("Error in formatted response: " + validated_response_string)

    # If valid, return result, if not search for more documentation and begin process again
    if validated_response["Valid"] == True:
        structured_response = f"""{{
    Question: {query_text}
    Response: {response_text}
    Sources: {sources}
}}"""
        return structured_response
    elif validated_response["Valid"] == False:
        ingest_rulebook(validated_response["Game"])
        if(current_depth < MAX_DEPTH):
            return query_rag(query_text, current_depth)
        return "I have failed. Forgive me."
    else:
        invalid_message = "No valid data was found to answer this question."
        return invalid_message


def main(): # Human Generated
    parser = argparse.ArgumentParser(description="Ask a question about the board-game PDFs.")
    parser.add_argument("query_text", help="The question to answer.")
    print(query_rag(parser.parse_args().query_text))


if __name__ == "__main__":
    main()
