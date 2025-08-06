from pinecone import Pinecone
from google import genai
from google.genai import types
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

PINECONE_INDEX_NAME = "rag" #TODO CHANGE THIS TO YOUR PINECONE INDEX NAME

class CareerAdviceRAG:
    def __init__(self, pin, gog):
        PINECONE_API_KEY = pin
        GOOGLE_API_KEY = gog
        pc = Pinecone(PINECONE_API_KEY)
        self.index = pc.Index(PINECONE_INDEX_NAME)

        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.chat = self.client.chats.create(model="gemini-2.0-flash")

        self.conversation_history = []

        self.model = SentenceTransformer('avsolatorio/GIST-large-Embedding-v0')

    def google_embed(self, text):
        result = self.client.models.embed_content(
                model="models/text-embedding-004", # text-embedding-large-exp-03-07
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY") #RETRIEVAL_QUERY, QUESTION_ANSWERING
        )
        return result.embeddings[0].values

    def gist_embed(self, text):
        return self.model.encode(text).tolist()

    def retrieveal(self, query: str, top_k: int = 1):
        """Retrieve the most relevant chunks from the vector database."""
        #query_embedding = self.google_embed(query)
        query_embedding = self.gist_embed(query)

        context = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        chunks = None

        return chunks, context

    def rephrase_query(self, history, query):
        #Prompt to update the query with the context of the previous messages
        #OPTIONAL: Play around with this prompt to see if you can improve it
        prompt = f"""You are an assistant tasked with taking a natural language \
        query from a user and converting it into a query for a vectorstore \
        Given a chat history and the latest user question which might reference \
        context in the chat history, formulate a standalone question which can \
        be understoon without the chat history. Do NOT answer the question, just \
        reformulate it if needed otherwise return it as is.
        Here is the chat history: {history}
        Here is the user query: {query}
        """

        response = self.client.models.generate_content(model="gemini-2.0-flash", contents=prompt)

        return response.text

    def generate_response(self, user_query: str):

        self.conversation_history.append({"role": "user", "content": user_query})

        #Rephrase the query. Before we send the query we want to repharse it to include context of the whole chat
        #This will allow more relavent retrievals.
        #This prompt has been written for you but you may edit it in the function above
        rephrased_query = self.rephrase_query(self.conversation_history, user_query)

        #Query the Pinecode database to get relvant transcript chunks
        #Change top_k to decide how many chunks to grab
        relevant_chunks, context = self.retrieveal(rephrased_query, top_k=3)

        #Final prompt that contains the user query/context.
        #YOU should add all your instructions here!
        final_prompt = f"""
        You are a career counselor that work with many youth in high school and
        college. Given a query inputted by a student, or youth looking for career
        advice, curate a relavent response grounded in the context information provided below
        to help guide the student to success. In the case that the context does not
        make sense for the query, explain to the student that you are not knowledgable
        in that realmn. Give a response with a rich story from the context embedded
        into it so that the student can have a meaningful takeaway.

        Mention the community members and also link and mention the sources at the end of the response.

        User Query: {user_query}
        Context: {context}
        """

        #Send the final prompt to the LLM
        response = self.chat.send_message(final_prompt).text
        self.conversation_history.append({"role": "assistant", "content": response})

        return relevant_chunks, response

    def get_conversation_history(self):
        """Return the current conversation history."""
        return self.conversation_history

    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation_history = []
