import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from toolbox_core import ToolboxSyncClient

# 1. Load the environment variables from .env
load_dotenv()

# 2. Model definition
# Use the Gemini 2.5 Flash model since it performs quickly and handles the processing well.
model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# 3. Configure short-term session to use the in-memory service
session_service = InMemorySessionService()

# 4. Read the instructions from a file in the same directory as this agent.py file.
script_dir = os.path.dirname(os.path.abspath(__file__))
instruction_file_path = os.path.join(script_dir, "agent-prompt.txt")
with open(instruction_file_path, "r") as f:
    instruction = f.read()

# 5. Set up the tools that we will be using for the root agent
toolbox_url = os.getenv("TOOLBOX_URL", "http://127.0.0.1:5000")
db_client = ToolboxSyncClient(toolbox_url)

get_accounts_tool = db_client.load_tool("get_accounts")
get_balance_tool = db_client.load_tool("get_balance")
get_transactions_tool = db_client.load_tool("get_transactions")
check_minimum_balance_tool = db_client.load_tool("check_minimum_balance")

# 6. Add the tools to the toolset
tools = [
    get_accounts_tool,
    get_balance_tool,
    get_transactions_tool,
    check_minimum_balance_tool,
]

# 7. Create the Agent
root_agent = Agent(
    name="deposit_agent",
    description="Agent that handles deposit accounts.",
    instruction=instruction,
    model=model,
    tools=tools,  # type: ignore
)
