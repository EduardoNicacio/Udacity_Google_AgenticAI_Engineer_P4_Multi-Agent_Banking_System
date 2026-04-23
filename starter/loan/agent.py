import os
from dotenv import load_dotenv
from google.adk import Runner
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from toolbox_core import ToolboxSyncClient
from .loan import loan_approval_workflow

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

# 5.a Loan tools
get_loan_balance_tool = db_client.load_tool("get_loan_balance")
get_loan_details_tool = db_client.load_tool("get_loan_details")
get_next_payment_date_tool = db_client.load_tool("get_next_payment_date")

# 6. Load the necessary sub-agents
loan_workflow = loan_approval_workflow

# 7. Create the agent with sub-agents and all the required tools
root_agent = Agent(
    name="loan_agent",
    description="Agent that handles loan applications.",
    instruction=instruction,
    sub_agents=[
        loan_workflow
    ],
    tools = [
        get_loan_balance_tool,
        get_loan_details_tool,
        get_next_payment_date_tool,
    ],
    model=model,
)

# 8. Defines the ADK Runner for the root_agent
runner = Runner(
    agent=root_agent,
    session_service=session_service,
    app_name="national_bank_loan_agent",
)
