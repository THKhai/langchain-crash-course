from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def get_current_time(*args, **kwargs):
    import datetime 
    now = datetime.datetime.now() 
    return now.strftime("%I:%M %p") 


tools = [
    Tool(
        name="Time",  # Name of the tool
        func=get_current_time,  # Function that the tool will execute
        # Description of the tool
        description="Useful for when you need to know the current time",
    ),
]

# https://smith.langchain.com/hub/hwchase17/react
prompt = hub.pull("hwchase17/react")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    # tùy chọn:
    # temperature=0.3,
    # max_output_tokens=1024,
)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
)

response = agent_executor.invoke({"input": "What time is it?"})

print("response:", response)
