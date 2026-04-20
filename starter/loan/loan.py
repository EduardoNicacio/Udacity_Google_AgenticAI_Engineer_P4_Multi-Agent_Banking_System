import logging
import os
from dotenv import load_dotenv
from typing import AsyncGenerator

from google.adk.agents import Agent, LlmAgent, SequentialAgent, ParallelAgent, BaseAgent, InvocationContext
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent, AGENT_CARD_WELL_KNOWN_PATH
from google.adk.events import Event, EventActions
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from toolbox_core import ToolboxSyncClient

from .datastore import datastore_search_tool

# 1. Load the environment variables from .env
load_dotenv()

# 2. Model definition
# Use the Gemini 2.5 Flash model since it performs quickly and handles the processing well.
model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# 3. Configure short-term session to use the in-memory service
session_service = InMemorySessionService()

# 4. Read the instructions from a file in the same directory as this agent.py file.
def load_instructions( prompt_file: str ):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    instruction_file_path = os.path.join( script_dir, prompt_file )
    with open(instruction_file_path, "r") as f:
        return f.read()
    
# 5. Set up the tools that we will be using for the root agent
toolbox_url = os.getenv("TOOLBOX_URL", "http://127.0.0.1:5000")
db_client = ToolboxSyncClient(toolbox_url)

# 6. Create loan_info_tool (stage 2)
get_loan_balance_tool = db_client.load_tool("get_loan_balance")

# 7. Connect to Deposit Agent via A2A
deposit_agent = RemoteA2aAgent(
    name="deposit",
    agent_card="http://localhost:8000/a2a/deposit/.well-known/a2a-agent-card.json"
)

# 8. Create agent that gets the requested loan value (stage 4)
get_requested_value_agent = LlmAgent(
    name="get_requested_value_agent",
    instruction="Extract the loan type and requested amount from the user's request.",
    output_key="loan_request_details",
    model=model
)

# 10. Define individual agents for parallel calling
outstanding_balance_agent = LlmAgent(
    name="outstanding_balance_agent",
    instruction="Get the outstanding loan balance for the user.",
    tools=[get_loan_balance_tool],
    output_key="outstanding_balance",
    model=model
)

policy_agent = LlmAgent(
    name="policy_agent",
    instruction="Check the loan policy documents for this type of loan.",
    tools=[datastore_search_tool],
    output_key="policy_criteria",
    model=model
)

user_profile_agent = LlmAgent(
    name="user_profile_agent",
    instruction="Check the customer profile document.",
    tools=[datastore_search_tool],
    output_key="customer_profile",
    model=model
)

# 11. Define the parallel agent for information gathering
info_gathering_agent = ParallelAgent(
    name="info_gathering",
    sub_agents=[outstanding_balance_agent, policy_agent, user_profile_agent]
)

# 12. Custom Agent for calculation (computes minimum required equity)
class TotalValueAgent(BaseAgent):

  def __init__(
    self,
    name: str,
  ):
    super().__init__(
      name=name,
    )

  async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
    # TODO: Implement
    yield None # type: ignore

# 13. Instantiate the TotalValueAgent to use (stage 4)
total_value_agent = TotalValueAgent(name="total_value_agent")

# 14. Create agent that checks user's equity (stage 4)
check_equity_agent = LlmAgent(
    name="check_equity_agent",
    instruction="Ask the deposit agent to check if the user has the minimum required equity. DO NOT ask for total balance.",
    sub_agents=[deposit_agent],
    output_key="equity_check_result",
    model=model
)

# 15. Final decision Agent (as per rubric -> safe responses)
final_decision_agent = LlmAgent(
    name="final_decision_agent",
    instruction="""Evaluate the equity check and customer rating.
    If approved: Mention a loan officer will contact them. 
    If rejected: Be polite and DO NOT reveal specific policy details or customer ratings.""",
    model=model
)

# 16. Create an Agent that uses these agents to approve or reject a new loan application
loan_approval_workflow = SequentialAgent(
    name="loan_approval_workflow",
    description="Workflow to approve or reject a new loan application.",
    sub_agents=[
        get_requested_value_agent,
        info_gathering_agent,
        total_value_agent,
        check_equity_agent,
        final_decision_agent
    ]
)