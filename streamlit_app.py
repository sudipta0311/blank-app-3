import streamlit as st
import os



# Retrieve secrets using st.secrets
# Add an environment variable
AZURE_OPENAI_API_KEY = st.secrets.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = st.secrets.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME= st.secrets.get("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION= st.secrets.get("AZURE_OPENAI_API_VERSION") 
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY") 

from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_endpoint= AZURE_OPENAI_ENDPOINT,
    azure_deployment= AZURE_OPENAI_DEPLOYMENT_NAME,
    openai_api_version= AZURE_OPENAI_API_VERSION,
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

pc = Pinecone(PINECONE_API_KEY)


# vector store 
index_name = "assistanttelco"

index = pc.Index(index_name)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

retriever = vector_store.as_retriever()

################################# AGENT STATE ###########################################

from typing import Annotated, Sequence
from typing_extensions import TypedDict
from typing import List

from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]
    metadata_filter: [dict]
    question: str
    generation: str
    documents: List[str]  


############################ GET QUESTION################################

from langchain import hub
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


################################# Question rewriter ###################################

def transform_query(state):
    """
    Transform the query to produce a better question optimized for vector store retrieval.

    Args:
        state (dict): The current graph state

    Returns:
        dict: Updated state with a re-phrased question
    """
    print("---TRANSFORM QUERY---")

    latest_question = get_latest_user_question(st.session_state.conversation)
    
    # Retrieve the last user question to check for context
    previous_questions = [msg[1] for msg in st.session_state.conversation if msg[0] == "user"]
    last_question = previous_questions[-2] if len(previous_questions) > 1 else ""
    
    # Use LLM to determine if the question is a follow-up
    follow_up_check_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant that determines if a question is a follow-up to a previous one. "
        "A follow-up question continues the context of the previous question "
        "and does not introduce a new product, topic, or category."),
        ("human", "Previous question: {last_question}"
        "\nNew question: {latest_question}\nIs this a follow-up question? Answer 'yes' or 'no'. "
        "Consider it 'no' if the new question shifts to a different product, topic, or category.")
    ])
    
    follow_up_checker = follow_up_check_prompt | llm | StrOutputParser()
    is_follow_up = follow_up_checker.invoke({"last_question": last_question, "latest_question": latest_question}).strip().lower() == "yes"
    
    if is_follow_up and last_question:
        combined_question = f"{last_question} Follow-up: {latest_question}"
    else:
        combined_question = latest_question
    
    # Question re-writer prompt
    system = """You are a virtual assistant specializing in YouSee Denmark. 
            Your job is to refine the user's question to be more specific to YouSee Denmark’s services, plans, network, or offers. 

            Additionally, you are a question re-writer that converts an input question into a better version optimized for vector store retrieval. 
            Analyze the input and reason about the underlying semantic intent or meaning to generate a more precise and relevant question. 

            Ensure that the rewritten question remains relevant to YouSee Denmark."""

    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question.")
    ])

    question_rewriter = re_write_prompt | llm | StrOutputParser()

    # Invoke re-writer
    better_question = question_rewriter.invoke({"question": combined_question})

    st.session_state.conversation.append(("user", better_question))
    state["question"] = better_question

    print("---New question---" + state["question"])
    return {"question": better_question}



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
    - accessories
    - ott
    - fixed internet 
    - 5g 
    for ex- if user asks about plans , subscriptions or mobile calling pack etc then metadata will be plans
    if user asked for streeming service then metadata will be TV 
    if user ask for 5g service , then metadata will be 5g
    if user asks for mobile phones then metadata will be phone
    if user asks for tablet, ipad then metadata will be tablet
    if user asks for smartwatch then metadata will be watch
    if user asks for any mobile accessories i.e headphone , charger , cover then metadata will be accessories 
    Additionally, the documents may have a 'brand' metadata field for filtering by brand.The possible brands are
    - Apple
    - Samsung
    - Google
    - Motorela
    - Xiaomi
    - Yousee 


    Instructions:
    - Identify the appropriate category based on the user's question.
    - If the question specifies a particular brand, include it in the metadata filter.
    - Return the metadata filter in JSON format.
    """

    question = state["question"]
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


############################ RETRIEVER ################################
from langchain.schema import Document

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")

    question = state["question"]
    get_metadata_extractor(state)  # Extract metadata

    metadata_filter = state.get("metadata_filter", None)  # Get metadata if available

    print("---METADATA---")
    print(metadata_filter if metadata_filter else "No metadata filtering applied")

    # Perform retrieval with or without metadata filtering
    if metadata_filter:
        documents = retriever.invoke(question, filter=metadata_filter)
    else:
        documents = retriever.invoke(question)

    print("---DOCUMENTS---")
    print(documents)

    return {"documents": documents, "question": question}


#############################################GRAGE###############################################

from typing import Annotated, Literal, Sequence

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

    documents = state["documents"]
    question = state["question"]

    print(" graded question "+ question)

    #print("retrived doc =" + docs)

    scored_result = chain.invoke({"question": question, "context": documents})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"

######################################### GENERATE##################################################################


def generate(state):
    print("---GENERATE---")
    messages = state["messages"]

    documents = state["documents"]
    question = state["question"]

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
    response = rag_chain.invoke({"context": documents, "question": question})
    print("---GENERATE DONE---")
    return {"messages": [response],"documents": documents, "question": question, "generation": response}


#############################################HALLUCINATION AND ANSWER TEST ###############################################

### Hallucination Grader


# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# LLM with function call
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader


### Answer Grader


# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# LLM with function call

structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Prompt
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        return "useful"
    
        '''
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
            '''
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


############################### GRAPH#######################################################

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages

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
    final_msg = ("Sorry, this question is beyond my knowledge, as a virtual assistant I can only assist you "
                 "with products for YS denmark")
    return {"messages": [AIMessage(content=final_msg)]}


# Define a new graph.
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve)  # retrieve
#workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("final_response", final_response)


# Build graph

workflow.add_edge(START, "transform_query")

workflow.add_edge("transform_query", "retrieve")

# In the retrieval branch, use the limited grade_documents function.
workflow.add_conditional_edges(
    "retrieve",
    grade_documents_limited,
    {
        "rewrite": "transform_query",
        "generate": "generate",
        "final": "final_response"
    }
)

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "generate",
    },
)

# Compile the graph.
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

#############################################GUI#################################################
import uuid
import streamlit as st
import time

# Generate a thread_id dynamically if it doesn't exist in session state.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Now use the dynamically generated thread_id in your config.
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# Initialize session states if they don’t exist
if "history" not in st.session_state:
    st.session_state.history = ""

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

            # Show "Assistant typing..." flashing in real-time
            typing_placeholder = st.empty()
            for _ in range(5):  # Flashing effect
                typing_placeholder.markdown("**Assistant typing...** ⏳")
                time.sleep(0.5)
                typing_placeholder.markdown("")
                time.sleep(0.5)

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

