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

load_dotenv()

def get_current_time(*args, **kwargs):
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")


def wikipedia_tool_func(query: Optional[str] = None, title: Optional[str] = None) -> str:
    topic = (query or title or "").strip()
    if not topic:
        return "Please provide a topic (query or title)."
    try:
        from wikipedia import summary, exceptions
        wikipedia.set_lang("en")
    except Exception:
        return "The 'wikipedia' package is not installed. Run: pip install wikipedia"
    try:
        return summary(topic, sentences=2, auto_suggest=True, redirect=True)
    except exceptions.DisambiguationError as e:
        opts = ", ".join(e.options[:5])
        return f"Topic is ambiguous. Try one of: {opts} ..."
    except Exception as e:
        return f"I couldn't find information on that. ({e.__class__.__name__})"

class TimeInput(BaseModel):
    pass


class WikipediaInput(BaseModel):
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

prompt = hub.pull("hwchase17/structured-chat-agent")

# Initialize a Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    system_instruction=(
        "You are an AI assistant that can use the tools Time and Wikipedia."
    ),
)

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)


agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)


agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,  
    handle_parsing_errors=True, 
)

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    response = agent_executor.invoke({"input": user_input})
    print("Bot:", response["output"])