from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AnyMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()

serper = GoogleSerperAPIWrapper()   
google_api_key = os.getenv("GOOGLE_API_KEY")

class State(TypedDict):
    messages:Annotated[list[AnyMessage], add_messages]

def do_search(title : str):
    """ Search for any details online for the given title
    Args:
        title: The title used for input to the search method, its a string value
    """
    print("do_search called")
    print(title)

    serper = GoogleSerperAPIWrapper()
    res = serper.run(title)
    return res


def find_topic(state:State):
    print("find_topic node has been called")
    title = state["messages"][-1]
    print(title.name)
    prompt = ""
    llm = ChatOpenAI(model="gpt-4o-mini")
    result = ""
    if title.name == None:
        prompt = f"Search for '{title}' using the tool and in the response provide me the list of topics for this title. If you don't find any answer from the tool them simple respond 'don't know'"
        llm_with_tools = llm.bind_tools([do_search])
        result = llm_with_tools.invoke(prompt)
    else:
        prompt = f"'{title}'From this given trending topics, pick one title and provide me title alone on the response"
        result = llm.invoke(prompt)
    return {"messages":[result]}

def topic_router(state:State):
    print("topic_router router has been called")
    last_message = state["messages"][-1]
    print(last_message)
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("Need to call tools")
        return "tools"
    else:
        return "write_script"
    

def write_script(state:State):
    message = state["messages"][-1]
    title = message.content
    print("Writing the script")
    print(title)
    prompt = f"""
        Title: {title}

        Write a short, engaging podcast script that explains a complex topic in a simple, conversational way. The topic can be from any field—science, tech, history, health, business, etc.

        Guidelines:

        Start with a relatable story or analogy to hook listeners.

        Break down complex ideas using plain language and vivid comparisons.

        Include a few:

        Fun facts or surprising insights

        Real-world examples or use cases

        Listener-friendly questions or reflections

        End with clear takeaways or something for the audience to think about.

        Tone: Friendly, curious, and easy to follow—like teaching a smart friend who’s new to the topic.
    """
    gemini_client = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=google_api_key)
    result = gemini_client.invoke(prompt)
    
    # Save the script content to script.md
    with open("script.md", "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(result.content)
    print(f"Script saved to script.md")
    
    return {"messages":[result]}


def buildGraph(graph_builder:StateGraph):
    graph_builder.add_node("topic_finder", find_topic)
    graph_builder.add_edge(START, "topic_finder")
    graph_builder.add_node("tools", ToolNode(tools=[do_search]))
    graph_builder.add_node("write_script", write_script)
    graph_builder.add_conditional_edges( "topic_finder", topic_router, {"tools": "tools", "write_script": "write_script"})
    graph_builder.add_edge("tools", "topic_finder")
    graph_builder.add_edge("write_script", END)

    graph = graph_builder.compile()
    graph_img = Image(graph.get_graph().draw_mermaid_png())
    display(graph_img)
    with open("graph_image.png", "wb") as f:
        f.write(graph_img.data)
    graph.invoke({"messages":"Find the number 1 trending topic on Agentic AI as of today, then write a podcast script for it"})


if __name__ == "__main__":
    graph_builder = StateGraph(State)
    buildGraph(graph_builder)

