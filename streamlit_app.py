from langchain_openai import OpenAIEmbeddings
import streamlit as st


import os

# Retrieve secrets using st.secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
LANGCHAIN_TRACING_V2 = st.secrets.get("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = st.secrets.get("LANGCHAIN_API_KEY")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")


# Load existing vector store

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

pc = Pinecone(api_key=PINECONE_API_KEY)


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# vector store 
index_name = "demoindex"

index = pc.Index(index_name)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

retriever = vector_store.as_retriever()

from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information abput telecom products.",
)

tools = [retriever_tool]

############################# Utility tasks ############################################

from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field


from langgraph.prebuilt import tools_condition

### Edges


def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade,method="function_calling")

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"


### Nodes


def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question contextualized for YOUSEE DENMARK.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with a re-phrased question specific to YOUSEE DENMARK
    """

    print("---TRANSFORM QUERY FOR YOUSEE DENMARK---")

    messages = state["messages"]
    question = messages[0].content  # Extract the user's question

    # Prompt to force contextualization for YOUSEE DENMARK
    msg = [
        HumanMessage(
            content=f"""
        You are a virtual assistant specializing in Yousee denmark.
        Your job is to refine the user's question to be more specific to Yousee Denmark’s services, plans, network, or offers.

        **User's Original Question:**
        {question}

        **Rewritten Question (must be relevant to Yousee denmark):**
        """,
        )
    ]

    # Invoke the model to rephrase the question with Airtel context
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    response = model.invoke(msg)

    return {"messages": [response]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
   # prompt = hub.pull("rlm/rag-prompt")
    prompt = PromptTemplate(
    template="""
    You are a telecom sales agent specializing in providing the best offers and plans for customers.
    Your goal is to assist customers by answering their questions, providing relevant information based on the available context,
    and creating a compelling sales proposal that convinces them to choose a product or service.

    **Context Information:**
    {context}

    **Customer's Question:**
    {question}

    **Instructions:**
    - If the context contains relevant details, use them to craft a persuasive sales pitch.
    - Highlight the key benefits, special offers, and why the customer should choose this product or service.
    - If there are multiple options, suggest the best one based on the customer's needs.
    - If no relevant information is available, politely inform the customer:
      "I'm sorry, but I don't have the details for that request at the moment."

    **Sales Proposal Format:**
    - **Greeting & Acknowledgment**: ("Thank you for your interest in our telecom services!")
    - **Personalized Offer**: ("Based on your query, here’s the best plan for you...")
    - **Key Benefits**: (Speed, coverage, price, special discounts, etc.)
    - **Call to Action**: ("This is a limited-time offer! Would you like to proceed with this?")
    """,
    input_variables=["context", "question"],)


    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


#print("*" * 20 + "Prompt[rlm/rag-prompt]" + "*" * 20)
#prompt = hub.pull("rlm/rag-prompt").pretty_print()  # Show what the prompt looks like

########################## graph #################################################################

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# Define AgentState without a retry_count field (since we'll use a global variable)
class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed.
    # Default is to replace; add_messages means "append."
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Global variable to track retry count.
global_retry_count = 0

# New wrapper to limit retries using a global variable.
def grade_documents_limited(state) -> str:
    global global_retry_count
    print("---TEST global retry count is ---", global_retry_count)
    
    decision = grade_documents(state)  # This function must be defined elsewhere.
    
    if decision == "rewrite":
        if global_retry_count >= 1:
            # Maximum retries reached: force to generate a response.
            print("---Maximum retries reached: switching to generate---")
            return "generate"
        else:
            # Increment the global retry counter and request a rewrite.
            global_retry_count += 1
            print("---after increment, global retry count is ---", global_retry_count)
            return "rewrite"
    else:
        return decision

# Define a new graph.
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between.
workflow.add_node("agent", agent)         # Agent node; function 'agent' must be defined.
retrieve = ToolNode([retriever_tool])       # 'retriever_tool' must be defined.
workflow.add_node("retrieve", retrieve)     # Retrieval node.
workflow.add_node("rewrite", rewrite)       # Rewriting the question; function 'rewrite' must be defined.
workflow.add_node("generate", generate)     # Generating the response; function 'generate' must be defined.

# Build the edges.
# Start by rewriting the query.
workflow.add_edge(START, "rewrite")
# After rewriting, call the agent node to decide whether to retrieve or not.
workflow.add_edge("rewrite", "agent")

# Conditional edge from the agent.
workflow.add_conditional_edges(
    "agent",
    # This condition uses tools_condition to decide if retrieval is needed.
    tools_condition,  # Function 'tools_condition' must be defined.
    {
        # If the agent decides to use tools, go to "retrieve".
        "tools": "retrieve",
        # Otherwise, end the graph.
        END: END,
    },
)

# After retrieval, use the limited grade_documents function to decide.
workflow.add_conditional_edges(
    "retrieve",
    grade_documents_limited,
)

# When generation is completed, end the graph.
workflow.add_edge("generate", END)
# Also allow looping: after rewriting, go back to agent (if needed).
workflow.add_edge("rewrite", "agent")

# Compile the graph.
# If you don't need a checkpointer, compile without it.
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
graph = workflow.compile()


#############################################GUI#################################################



config = {"configurable": {"thread_id": "aaa1234"}}

import streamlit as st
import pprint


# Define the Streamlit app
def run_virtual_assistant():
    st.title("Virtual Assistant")

    # Ask for user input
    user_input = st.text_input("Ask me anything about your telcom need")

    if user_input:
        # Prepare the input for the graph
        inputs = {
            "messages": [
                ("user", user_input),
            ]
        }

        # Initialize a variable to store the final message content
        final_message_content = ""

        # Process the input through the graph (assumes 'graph' and 'config' are defined globally)
        for output in graph.stream(inputs, config):
            for key, value in output.items():
                # Check if the value is a dict containing messages
                if isinstance(value, dict) and "messages" in value:
                    for msg in value["messages"]:
                        # Check if the message object has the 'content' attribute
                        if hasattr(msg, "content"):
                            final_message_content = msg.content + "\n"
                        else:
                            final_message_content = str(msg) + "\n"

        # Use st.markdown to render the content with preserved newlines
        st.markdown(final_message_content)

# Run the app
if __name__ == "__main__":
    run_virtual_assistant()