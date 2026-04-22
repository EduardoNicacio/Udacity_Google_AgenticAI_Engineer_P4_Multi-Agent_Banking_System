import logging
import os
from dotenv import load_dotenv
from google.adk import Runner
from google.adk.agents import Agent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent, AGENT_CARD_WELL_KNOWN_PATH
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

# 6. Loads all the available tools
# 6.a. Deposit tools
get_accounts_tool = db_client.load_tool("get_accounts")
get_balance_tool = db_client.load_tool("get_balance")
get_transactions_tool = db_client.load_tool("get_transactions")
check_minimum_balance_tool = db_client.load_tool("check_minimum_balance")

# 6.b. Loan tools
get_loan_balance_tool = db_client.load_tool("get_loan_balance")
get_loan_details_tool = db_client.load_tool("get_loan_details")
get_next_payment_date_tool = db_client.load_tool("get_next_payment_date")

# 7. Loads the Deposit Agent via A2A
deposit_agent = RemoteA2aAgent(
    name="deposit",
    agent_card="http://localhost:8001/a2a/deposit/.well-known/agent-card.json"
)

# 8. Loads the Loan Agent via A2A
loan_agent = RemoteA2aAgent(
    name="loan", 
    agent_card="http://localhost:8000/a2a/loan/.well-known/agent-card.json"
)

# 9. Set up other agents that we can delegate to
sub_agents=[
    deposit_agent,
    loan_agent,
]

# 10. Creates the agent with sub-agents and all the required tools
root_agent = Agent(
    name="manager_agent",
    description="Agent that handles loan applications.",
    instruction=instruction,
    sub_agents=sub_agents,
    tools=[
        get_accounts_tool,
        get_balance_tool,
        get_transactions_tool,
        check_minimum_balance_tool,
        get_loan_balance_tool,
        get_loan_details_tool,
        get_next_payment_date_tool,
    ],
    model=model,
)

# 11. Defines the ADK Runner for the root_agent
runner = Runner(
    agent=root_agent,
    session_service=session_service,
    app_name="national_bank_manager_agent",
)
