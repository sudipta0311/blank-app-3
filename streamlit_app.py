import getpass
import os

import streamlit as st

# Retrieve secrets using st.secrets
# Add an environment variable
AZURE_OPENAI_API_KEY = st.secrets.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = st.secrets.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME= st.secrets.get("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION= st.secrets.get("AZURE_OPENAI_API_VERSION") 


from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    openai_api_version=AZURE_OPENAI_API_VERSION,
)

import getpass
import os


from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
    azure_deployment='text-embedding-3-large',
    openai_api_version='2023-05-15',
)

################################# VECTOR STORE ###########################################


# Load existing vector store

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

pc = Pinecone('pcsk_2yWxfV_RzZcenPUjLkzMK78P8D2MEX6yfzSZJ2GYCKCfkiHUpgbj8ekG4yWfue7JJsEYtr')


#embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# vector store 
index_name = "assistanttelco"

index = pc.Index(index_name)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)


# Create retriever with metadata filtering support (optional)
retriever = vector_store.as_retriever(
     search_kwargs={"filter": None}  # Initially set to None, updated dynamically
)

from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "get_product_info",
    "Retrieve yousee products offers from knowlodge base based on  given query.",
)

tools = [retriever_tool]


########################### METADATA Identify ##########################################

from typing import Sequence, Annotated, TypedDict, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate


# Define MetadataFilter separately
class MetadataFilter(BaseModel):
    category: Optional[str] = Field(
        None,
        description="Category of the item (phone, plans, accessories, TV, fixed internet)."
    )
    brand: Optional[str] = Field(
        None,
        description="Brand name if specified (e.g., Samsung, Apple)."
    )

# Define MetadataQuery using MetadataFilter
class MetadataQuery(BaseModel):
    metadata_filter: MetadataFilter = Field(
        ...,
        description="Metadata filter containing category and brand."
    )

    class Config:
        json_schema_extra = {
            "required": ["metadata_filter"]
        }
def get_metadata_extractor(state):
    """Extracts metadata from a user's question and updates the state."""

    # LLM with function call
    structured_llm_metadata = llm.with_structured_output(MetadataQuery, method="function_calling")

    # System prompt
    system = """
    You are an expert at identifying the correct metadata filters for a search query in a vector store. 
    The vector store contains documents categorized under:
    - phone
    - tablet
    - watch
    - plans
    - subscription
    - accessories
    - ott
    - fixed internet 
    - 5g
    Additionally, the documents may have a 'brand' metadata field for filtering by brand.

    Instructions:
    - Identify the appropriate category based on the user's question.
    - If the question specifies a particular brand, include it in the metadata filter.
    - Return the metadata filter in JSON format.
    """

    get_latest_user_question(st.session_state.conversation)
    metadata_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    metadata_extractor = metadata_prompt | structured_llm_metadata

    # Extract metadata
    extracted_metadata = metadata_extractor.invoke({"question": state["messages"][-1].content})

    # Filter out None values from the extracted metadata
    metadata_dict = extracted_metadata.metadata_filter.model_dump()
    filtered_metadata = {k: v for k, v in metadata_dict.items() if v is not None}

    # Update state based on whether metadata was found
    if filtered_metadata:
        state["metadata_filter"] = filtered_metadata  # Override with new metadata
    elif "metadata_filter" in state:
        del state["metadata_filter"]  # Remove old metadata if no new metadata found

    return {"metadata_filter": [filtered_metadata]} if filtered_metadata else {}



############################# Utility tasks ############################################
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]
    metadata_filter: [dict]

from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field


from langgraph.prebuilt import tools_condition

def get_latest_user_question(messages):
    # Iterate over the messages in reverse order
    for role, content in reversed(messages):
        if role.lower() == "user":
            return content
    return ""

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
    model = llm

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

    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    #question = messages[0].content
    #question = get_latest_user_question(messages)
    question = get_latest_user_question(st.session_state.conversation)

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
     # Extract metadata dynamically
    get_metadata_extractor(state)
    metadata_filter = state.get("metadata_filter", None)

    print("---metadata----")
    print(metadata_filter)


    # Summarizing long messages before sending them to the model
    def summarize_message(msg):
        if isinstance(msg, AIMessage) and len(msg.content) > 1000:
            return AIMessage(content=msg.content[:997] + "...")
        return msg

    messages = [summarize_message(msg) for msg in messages]

    # Bind tools dynamically with metadata-aware retriever
    
    global retriever

    retriever = vector_store.as_retriever(
    search_kwargs = {"filter": metadata_filter} if metadata_filter else {}  #
    )

    
    model=llm
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list

    return {"messages": [response]}

def rewrite(state):
    """
    Transform the query to produce a better question contextualized for YOUSEE DENMARK,
    considering follow-up questions and previous queries.
    
    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with a re-phrased question specific to YOUSEE DENMARK
    """
    print("---TRANSFORM QUERY FOR YOUSEE DENMARK---")
    
    messages = state["messages"]
    latest_question = get_latest_user_question(st.session_state.conversation)
    
    
    # Retrieve the last user question to check for context
    previous_questions = [msg[1] for msg in st.session_state.conversation if msg[0] == "user"]
    last_question = previous_questions[-2] if len(previous_questions) > 1 else ""
    
    # Determine if the new question is a follow-up
    follow_up_indicators = ["price", "tell", "what", "explain more", "how", "and?","where"]
    is_follow_up = any(indicator in latest_question.lower() for indicator in follow_up_indicators)
    
    if is_follow_up and last_question:
        combined_question = f"{last_question} Follow-up: {latest_question}"
    else:
        combined_question = latest_question

    
    # Prompt to force contextualization for YouSee Denmark
    msg = [
        HumanMessage(
            content=f"""
            You are a virtual assistant specializing in YouSee Denmark.
            Your job is to refine the user's question to be more specific to YouSee Denmark’s services, plans, network, or offers.
            
            **User's Original Question:**
            {latest_question}
            
            **Rewritten Question (must be relevant to YouSee Denmark):**
            """,
        )
    ]
    
    # Invoke the model to rephrase the question
    model = llm
    response = model.invoke(msg)
    
    print("Relevant contextualized question=" + response.content)
    return {"messages": [response]}


def generate(state):
    print("---GENERATE---")
    messages = state["messages"]

    #question = get_latest_user_question(messages)
    question = get_latest_user_question(st.session_state.conversation)
    # Assume the last assistant message (or retrieved content) holds the context.
    last_message = messages[-1]
    docs = last_message.content

    prompt = PromptTemplate(
        template="""You are a telecom sales agent specializing in providing the best offers and plans for customers.
        Your goal is to assist customers by answering their questions, providing relevant information based on the available context,
        and creating a compelling sales proposal that convinces them to choose a product or service.
        
        **Context Information:**
        {context}
        **Customer's Question:**
        {question}
        
        **Instructions:**
        - If the context contains relevant details, use them to craft a persuasive sales pitch.
        - Highlight the key benefits, special offers, and why the customer should choose this product or service.
        - If no relevant information is available, politely inform the customer:
          "I'm sorry, but I don't have the details for that request at the moment."
        """,
        input_variables=["context", "question"],
    )

   # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

################################# GRAPH##################################
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
import streamlit as st

# Initialize session state for conversation history if it doesn't exist.
if "conversation" not in st.session_state:
    st.session_state.conversation = []  # List of tuples like ("user", "question") or ("assistant", "response")
    # Initialize session state for retry count.
if "retry_count" not in st.session_state:
    st.session_state.retry_count = 0

# New wrapper to limit retries using session state.
def grade_documents_limited(state) -> str:
    # Use the retry count from session state



    decision = grade_documents(state)  # This function must be defined elsewhere.
    retry_count = st.session_state.retry_count +1
    print("---TEST retry count is ---", retry_count)

    if decision == "rewrite":
        if retry_count >= 1:
            # Maximum retries reached: return a special decision "final"
            print("---Maximum retries reached: switching to final response---")
            return "final"
        else:
            # Increment the retry counter in session state.
            st.session_state.retry_count = retry_count + 1
            print("---after increment, retry count is ---", st.session_state.retry_count)
            return "rewrite"
    else:
        return decision
    
    # New node to handle the final response.
def final_response(state):
    final_msg = ("Sorry, this question is beyond my knowledge, ask me about any other question on ys products ")
    return {"messages": [AIMessage(content=final_msg)]}

# Define a new graph.
workflow = StateGraph(AgentState)

# Define the nodes (agent, retrieve, rewrite, generate, and final_response).
workflow.add_node("agent", agent)         # Agent node; function 'agent' must be defined.
retrieve = ToolNode([retriever_tool])       # 'retriever_tool' must be defined.
workflow.add_node("retrieve", retrieve)     # Retrieval node.
workflow.add_node("rewrite", rewrite)       # Rewriting the question; function 'rewrite' must be defined.
workflow.add_node("generate", generate)     # Generating the response; function 'generate' must be defined.
workflow.add_node("final_response", final_response)  # Final response node.

# Build the edges.
workflow.add_edge(START, "rewrite")
workflow.add_edge("rewrite", "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,  # Function 'tools_condition' must be defined.
    {
        "tools": "retrieve",
        END: END,
    },
)
# In the retrieval branch, use the limited grade_documents function.
workflow.add_conditional_edges(
    "retrieve",
    grade_documents_limited,
    {
        "rewrite": "rewrite",
        "generate": "generate",
        "final": "final_response"
    }
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile the graph.
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

#############################################GUI#################################################
import uuid
import streamlit as st

# Generate a thread_id dynamically if it doesn't exist in session state.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Now use the dynamically generated thread_id in your config.
config = {"configurable": {"thread_id": st.session_state.thread_id}}

if "history" not in st.session_state:
    st.session_state.history = ""

# Initialize session state for conversation history if it doesn't exist.
if "conversation" not in st.session_state:
    st.session_state.conversation = []  # List of tuples like ("user", "question") or ("assistant", "response")

def run_virtual_assistant():
    st.title("Virtual Agent")

    # Display conversation history if available.
    if st.session_state.conversation:
        with st.expander("Click here to see the old conversation"):
            st.subheader("Conversation History")
            st.markdown(st.session_state.history)

    # Use a form to handle user input and clear the field after submission.
    with st.form(key="qa_form", clear_on_submit=True):
        user_input = st.text_input("Ask me anything about telco offers (or type 'reset' to clear):")
        submit_button = st.form_submit_button(label="Submit")

    if submit_button and user_input:
        # Allow the user to reset the conversation.
        if user_input.strip().lower() == "reset":
            st.session_state.conversation = []
            st.session_state.history = ""
            st.experimental_rerun()
        else:
            # Append the user's question to the conversation history.
            st.session_state.conversation.append(("user", user_input))
            st.session_state.retry_count = 0

            # Prepare the input for the graph using the entire conversation history.
            inputs = {"messages": st.session_state.conversation}
            
            # Display "Assistant typing..."
            typing_placeholder = st.empty()
            typing_placeholder.markdown("**Assistant typing...⏳**")

            final_message_content = ""
            
            # Process the input through the graph (assumes 'graph' is defined globally).
            for output in graph.stream(inputs, config):
                for key, value in output.items():
                    # Check if the value is a dict containing messages.
                    if isinstance(value, dict) and "messages" in value:
                        for msg in value["messages"]:
                            if hasattr(msg, "content"):
                                final_message_content = msg.content + "\n"
                                # Append the assistant response to conversation history.
                                st.session_state.conversation.append(("assistant", msg.content))
                            else:
                                final_message_content = str(msg) + "\n"
                                st.session_state.conversation.append(("assistant", str(msg)))

            # Clear "Assistant typing..." and display the response
            typing_placeholder.empty()
            st.markdown(final_message_content)
            st.session_state.history += "################MESSAGE###############\n" + final_message_content

if __name__ == "__main__":
    run_virtual_assistant()
