import os
from dotenv import load_dotenv
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

# 6. Load the necessary Agents
# 6.a. Deposit Agent
deposit_agent = RemoteA2aAgent(
    name="deposit",
    agent_card="http://localhost:8000/a2a/deposit/.well-known/a2a-agent-card.json"
)

# 6.b. Loan Agent
loan_agent = RemoteA2aAgent(
    name="loan", 
    agent_card="http://localhost:8000/a2a/loan/.well-known/a2a-agent-card.json"
)

# 7. Create the agent with sub-agent and all the required tools
root_agent = Agent(
    name="loan_agent",
    description="Agent that handles loan applications.",
    instruction=instruction,
    sub_agents=[
        deposit_agent, 
        loan_agent,
    ],
    model=model,
)
