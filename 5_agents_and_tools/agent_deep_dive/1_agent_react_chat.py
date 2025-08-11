from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool
from typing import Optional


# Load environment variables from .env file
load_dotenv()


# Define Tools
def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")


def wikipedia_tool_func(query: Optional[str] = None, title: Optional[str] = None) -> str:
    """Lookup a topic on Wikipedia by query or title and return a short summary."""
    topic = (query or title or "").strip()
    if not topic:
        return "Please provide a topic (query or title)."
    try:
        from wikipedia import summary, exceptions  # pip install wikipedia
    except Exception:
        return "The 'wikipedia' package is not installed. Run: pip install wikipedia"
    try:
        # auto_suggest/redirect giúp “bắt” đúng trang nếu tên hơi lệch
        return summary(topic, sentences=2, auto_suggest=True, redirect=True)
    except exceptions.DisambiguationError as e:
        opts = ", ".join(e.options[:5])
        return f"Topic is ambiguous. Try one of: {opts} ..."
    except Exception as e:
        return f"I couldn't find information on that. ({e.__class__.__name__})"

class TimeInput(BaseModel):
    """No input needed for Time."""
    pass


class WikipediaInput(BaseModel):
    # Cho phép cả hai khóa
    query: Optional[str] = Field(None, description="Topic to look up on Wikipedia")
    title: Optional[str] = Field(None, description="Alias for query (topic title)")

time_tool = StructuredTool.from_function(
    name="Time",
    func=get_current_time,
    description="Useful for when you need to know the current time.",
    args_schema=TimeInput,
)

wikipedia_tool = StructuredTool.from_function(
    name="Wikipedia",
    func=wikipedia_tool_func,
    description=(
        "Look up a topic on Wikipedia. Accepts either {\"query\": \"...\"} "
        "or {\"title\": \"...\"}."
    ),
    args_schema=WikipediaInput,
)

tools = [time_tool, wikipedia_tool]
# Define the tools that the agent can use
# tools = [
#     Tool(
#         name="Time",
#         func=get_current_time,
#         description="Useful for when you need to know the current time.",
#     ),
#     Tool(
#         name="Wikipedia",
#         func=search_wikipedia,
#         description="Useful for when you need to know information about a topic.",
#     ),
# ]

# Load the correct JSON Chat Prompt from the hub
prompt = hub.pull("hwchase17/structured-chat-agent")

# Initialize a Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    system_instruction=(
        "You are an AI assistant that can use the tools Time and Wikipedia."
    ),
)

# Create a structured Chat Agent with Conversation Buffer Memory
# ConversationBufferMemory stores the conversation history, allowing the agent to maintain context across interactions
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

# create_structured_chat_agent initializes a chat agent designed to interact using a structured prompt and tools
# It combines the language model (llm), tools, and prompt to create an interactive agent
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# AgentExecutor is responsible for managing the interaction between the user input, the agent, and the tools
# It also handles memory to ensure context is maintained throughout the conversation
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,  # Use the conversation memory to maintain context
    handle_parsing_errors=True,  # Handle any parsing errors gracefully
)

# Initial system message to set the context for the chat
# SystemMessage is used to define a message from the system to the agent, setting initial instructions or context
# initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: Time and Wikipedia."
# memory.chat_memory.add_message(SystemMessage(content=initial_message))

# Chat Loop to interact with the user
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    # Add the user's message to the conversation memory
    # memory.chat_memory.add_message(HumanMessage(content=user_input))

    # Invoke the agent with the user input and the current chat history
    response = agent_executor.invoke({"input": user_input})
    print("Bot:", response["output"])

    # Add the agent's response to the conversation memory
    # memory.chat_memory.add_message(AIMessage(content=response["output"]))
