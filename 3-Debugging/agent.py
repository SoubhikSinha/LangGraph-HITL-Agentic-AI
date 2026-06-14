# Let's try to get the graph shown on LangGraph Studio

from langgraph.prebuilt import tools_condition
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import END, START
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Groq and Langsmith environment variables
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "TestProject"

# Initialize the LLM
from langchain.chat_models import init_chat_model
llm = init_chat_model("groq:llama-3.1-8b-instant")

# Creating a StateGraph
class State(TypedDict):
  messages:Annotated[list[BaseMessage], add_messages]


def make_tool_graph():
  # Creating the StateGraph with a Tool Call
  @tool
  def add(a:float, b:float):
    """Add 2 numbers"""
    return a + b
  tools = [add]
  tool_node = ToolNode([add])

  # Binding the tool with the LLM
  llm_with_tool = llm.bind_tools([add])

  # Node definition
  def call_llm_model(state:State):
    return {"messages":[llm_with_tool.invoke(state['messages'])]}

  # Graph
  builder=StateGraph(State)
  builder.add_node("tool_calling_llm", call_llm_model)
  builder.add_node("tools", ToolNode(tools))

  # Adding Edges
  builder.add_edge(START, "tool_calling_llm")
  builder.add_conditional_edges (
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition
  )

  builder.add_edge("tools", END)
  '''
  builder.add_edge("tools", tool_calling_llm) --> This will get you the AI Message in the output / response
  '''

  # Compiling the Graph
  graph = builder.compile()

  return graph

# Initializing the Graph
tool_agent = make_tool_graph()