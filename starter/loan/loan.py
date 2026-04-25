import json
import os
from dotenv import load_dotenv
from typing import AsyncGenerator, Any

from google.adk import Runner
from google.adk.agents import (
    LlmAgent,
    SequentialAgent,
    ParallelAgent,
    BaseAgent,
    InvocationContext,
)
from google.adk.agents.remote_a2a_agent import (
    RemoteA2aAgent,
    AGENT_CARD_WELL_KNOWN_PATH,
)
from google.adk.events import Event, EventActions
from google.adk.sessions import InMemorySessionService

from pydantic import BaseModel, Field
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
def load_instructions(prompt_file: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    instruction_file_path = os.path.join(script_dir, prompt_file)
    with open(instruction_file_path, "r") as f:
        return f.read()


# 5. Set up the tools that we will be using for the root agent
toolbox_url = os.getenv("TOOLBOX_URL", "http://127.0.0.1:5000")
db_client = ToolboxSyncClient(toolbox_url)

# 6. Create loan_info_tool (stage 2)
get_loan_balance_tool = db_client.load_tool("get_loan_balance")

# 7. Connects to the Deposit Agent via A2A
deposit_agent = RemoteA2aAgent(
    name="deposit",
    agent_card=f"http://127.0.0.1:8000/a2a/deposit{AGENT_CARD_WELL_KNOWN_PATH}",
    use_legacy=False,
)



# 8. Create agent that gets the requested loan value (stage 4)
class LoanRequestDetails(BaseModel):
    """Schema for loan request details"""
    loan_type: str = Field(..., description="Type of loan requested (auto/personal/home/improvement/recreational)")
    loan_amount: float | None = Field(None, description="Amount requested in USD")
    currency: str = "USD"

get_requested_value_agent = LlmAgent(
    name="get_requested_value_agent",
    description="Agent that gets the requested loan value",
    instruction=load_instructions("loan-request-prompt.txt"),
    output_schema=LoanRequestDetails,
    output_key="loan_request_details",
    model=model,
)

# 10. Define individual agents for parallel calling
# 10.a. Oustanding Balance Agent
class OutstandingBalance(BaseModel):
    """Schema for outstanding balance"""
    outstanding_balance: float = Field(..., description="Total outstanding loan balance in USD")
    currency: str = "USD"

outstanding_balance_agent = LlmAgent(
    name="outstanding_balance_agent",
    description="Agent that checks for outstanding balances during the Loan workflow",
    instruction=load_instructions("outstanding-balance-prompt.txt"),
    tools=[get_loan_balance_tool],
    output_schema=OutstandingBalance,
    output_key="outstanding_balance",
    model=model,
)

# 10.b. Loan Policy Agent
class DebtToEquityRatios(BaseModel):
    """Schema for debt-to-equity ratios"""
    auto_loans_under_10k: float = Field(..., description="DE ratio for auto loans under $10k")
    auto_loans_10k_plus: float = Field(..., description="DE ratio for auto loans $10k+")
    recreational_vehicles: float = Field(..., description="DE ratio for recreational vehicles")
    home_improvement_under_20k: float = Field(..., description="DE ratio for home improvement under $20k")
    home_improvement_20k_plus: float = Field(..., description="DE ratio for home improvement $20k+")

class CustomerRatingRequirements(BaseModel):
    """Schema for customer rating requirements"""
    auto_loans_under_10k: list[str] = Field(..., description="Acceptable ratings for auto loans under $10k")
    auto_loans_10k_plus: list[str] = Field(..., description="Acceptable ratings for auto loans $10k+")
    recreational_vehicles: list[str] = Field(..., description="Acceptable ratings for recreational vehicles")
    home_improvement_under_20k: list[str] = Field(..., description="Acceptable ratings for home improvement under $20k")
    home_improvement_20k_plus: list[str] = Field(..., description="Acceptable ratings for home improvement $20k+")

class PolicyCriteria(BaseModel):
    """Schema for policy criteria"""
    debt_to_equity_ratios: DebtToEquityRatios = Field(..., description="Debt-to-equity ratios by loan type")
    customer_rating_requirements: CustomerRatingRequirements = Field(..., description="Acceptable credit ratings by loan type")
    policy_document_loaded: bool = True
    source_uri: str = ""

policy_agent = LlmAgent(
    name="policy_agent",
    description="Agent that checks the loan policy documents for this type of loan",
    instruction=load_instructions("policy-agent-prompt.txt"),
    tools=[datastore_search_tool],
    output_schema=PolicyCriteria,
    output_key="policy_criteria",
    model=model,
)

# 10.c. User Profile Agent
class CustomerProfile(BaseModel):
    """Schema for customer profile"""
    customer_id: int | None = Field(None, description="Customer ID from document/chat")
    credit_rating: str | None = Field(None, description="Credit rating (Poor/Fair/Good/Great/Excellent)")
    financial_history_summary: str | None = Field(None, description="Summary of financial history")
    profile_loaded: bool = True
    other_profile_data: dict[str, Any] = {}

CustomerProfile.model_rebuild()

user_profile_agent = LlmAgent(
    name="user_profile_agent",
    description="Agent that analyzes the user profile when a Loan request is submitted",
    instruction=load_instructions("user-profile-base-prompt.txt"),
    tools=[datastore_search_tool],
    output_schema=CustomerProfile,
    output_key="customer_profile",
    model=model,
)

# 11. Define the parallel agent for information gathering
info_gathering_agent = ParallelAgent(
    name="info_gathering",
    sub_agents=[outstanding_balance_agent, policy_agent, user_profile_agent],
)


# 12. Custom Agent for calculation (computes minimum required equity)
class TotalValueState(BaseModel):
    """Schema for total value agent state"""
    minimum_deposits_required: float = Field(..., description="Minimum deposit balance required")
    customer_rating_requirements: dict[str, list[str]] = Field(..., description="Rating requirements by loan type")
    de_ratio_used: float = Field(..., description="Debt-to-equity ratio used in calculation")
    total_debt: float = Field(..., description="Total debt if loan is approved")

class TotalValueAgent(BaseAgent):
    """
    Calculates the minimum required deposit balance based on loan amount,
    outstanding balance, and policy debt-to-equity ratio.

    Formula from policy document:
    - Minimum Deposits Required = Total Debt / DE Ratio

    Where Total Debt = Existing Debt + New Loan Request
    """

    def __init__(self, name: str):
        super().__init__(name=name)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Calculate minimum required deposit balance based on loan amount and
        outstanding balance, reading all inputs from session state.
        """

        # 1. Read all inputs from session state
        state = ctx.session.state

        loan_request_raw = state.get("loan_request_details", {})
        outstanding_raw = state.get("outstanding_balance", 0)
        policy_raw = state.get("policy_criteria", "{}")

        # 2. Safely parse loan_request_details (LlmAgent output is a string)
        if isinstance(loan_request_raw, str):
            try:
                loan_request = json.loads(
                    loan_request_raw.strip().strip("```json").strip("```")
                )
            except (json.JSONDecodeError, AttributeError):
                loan_request = {}
        else:
            loan_request = (
                loan_request_raw if isinstance(loan_request_raw, dict) else {}
            )

        loan_amount_requested = float(loan_request.get("loan_amount", 0))
        loan_type = str(loan_request.get("loan_type", "personal")).lower()

        # 3. Safely parse outstanding_balance (may be a string like "$5,000" or a number)
        if isinstance(outstanding_raw, dict):
            outstanding_raw = outstanding_raw.get("outstanding_balance", 0)
        elif isinstance(outstanding_raw, str):
            try:
                parsed = json.loads(outstanding_raw)
                if isinstance(parsed, dict):
                    outstanding_raw = parsed.get("outstanding_balance", 0)
                else:
                    outstanding_raw = parsed
            except ValueError:
                outstanding_raw = (
                    outstanding_raw.replace("$", "").replace(",", "").strip()
                )

        outstanding_balance = float(outstanding_raw)

        # 4. Safely parse policy_criteria (LlmAgent output is a string — expect JSON)
        if isinstance(policy_raw, str):
            try:
                policy_criteria = json.loads(
                    policy_raw.strip().strip("```json").strip("```")
                )
            except (json.JSONDecodeError, AttributeError):
                policy_criteria = {}
        else:
            policy_criteria = policy_raw if isinstance(policy_raw, dict) else {}

        # 5. Determine the correct DE ratio from parsed policy
        ratios = policy_criteria.get("debt_to_equity_ratios", {})
        debt_to_equity_ratio = self._resolve_de_ratio(
            loan_type, loan_amount_requested, ratios
        )

        # 6. Calculate minimum deposits required
        total_debt = outstanding_balance + loan_amount_requested

        if debt_to_equity_ratio > 0:
            minimum_deposit_required = round(total_debt / debt_to_equity_ratio, 2)
        else:
            minimum_deposit_required = 0.0

        # Extract rating requirements to forward downstream
        customer_rating_requirements = policy_criteria.get(
            "customer_rating_requirements", {}
        )

        # 7. Write result to session state via EventActions.state_delta
        yield Event(
            author=self.name,
            actions=EventActions(
                state_delta={
                    "minimum_deposits_required": str(minimum_deposit_required),
                    "customer_rating_requirements": customer_rating_requirements,
                    "de_ratio_used": str(debt_to_equity_ratio),
                    "total_debt": str(total_debt),
                }
            ),
        )

    @staticmethod
    def _resolve_de_ratio(loan_type: str, amount: float, ratios: dict) -> float:
        """
        Select the correct debt-to-equity ratio based on loan type and amount.
        Returns a conservative default of 1.0 if no matching ratio is found.
        """
        if not ratios:
            return 1.0

        if loan_type.startswith("auto"):
            if amount < 10_000 and "auto_loans_under_10k" in ratios:
                return float(ratios["auto_loans_under_10k"])
            elif "auto_loans_10k_plus" in ratios:
                return float(ratios["auto_loans_10k_plus"])

        if "vehicle" in loan_type and "recreational_vehicles" in ratios:
            return float(ratios["recreational_vehicles"])

        if "home" in loan_type or "improvement" in loan_type:
            if amount < 20_000 and "home_improvement_under_20k" in ratios:
                return float(ratios["home_improvement_under_20k"])
            elif "home_improvement_20k_plus" in ratios:
                return float(ratios["home_improvement_20k_plus"])

        if loan_type.startswith("personal"):
            if amount < 100 and "personal_loans_under_100" in ratios:
                return float(ratios["personal_loans_under_100"])
            elif amount <= 500 and "personal_loans_100_to_500" in ratios:
                return float(ratios["personal_loans_100_to_500"])
            elif amount <= 5_000 and "personal_loans_500_to_5000" in ratios:
                return float(ratios["personal_loans_500_to_5000"])
            elif "personal_loans_5000_plus" in ratios:
                return float(ratios["personal_loans_5000_plus"])

        return 1.0


# Instantiate the TotalValueAgent with policy-aware logic
total_value_agent = TotalValueAgent(name="total_value_agent")

# 14. Create agent that checks user's equity (stage 4)
class EquityCheckResult(BaseModel):
    """Schema for equity check result"""
    meets_requirement: bool = Field(..., description="Whether deposits meet minimum requirement")

check_equity_agent = LlmAgent(
    name="check_equity_agent",
    description="Agent that checks the user's equity",
    instruction=load_instructions("check-equity-prompt.txt"),
    sub_agents=[deposit_agent],
    output_schema=EquityCheckResult,
    output_key="equity_check_result",
    model=model,
)

# 15. Final decision Agent (as per rubric -> safe responses)
class FinalDecision(BaseModel):
    """Schema for final decision"""
    approved: bool = Field(..., description="Whether loan is approved")
    reason_summary: str | None = Field(None, description="Brief summary of decision factors")

final_decision_agent = LlmAgent(
    name="final_decision_agent",
    description="The Final decision Agent",
    instruction=load_instructions("approval-report-prompt.txt"),
    output_schema=FinalDecision,
    output_key="final_decision_result",
    model=model,
)

# 16. Create an Agent that uses these agents to approve or reject a new loan application
loan_approval_workflow = SequentialAgent(
    name="loan_approval_workflow",
    description="Workflow to approve or reject a new loan application.",
    sub_agents=[
        get_requested_value_agent,  # Extracts loan amount and type
        info_gathering_agent,  # Parallel agent gathering all data
        total_value_agent,  # Calculates minimum deposit required using DE ratio from policy
        check_equity_agent,  # Verifies deposits meet requirement via A2A
        final_decision_agent,  # Makes final decision
    ],
)

# Defines the ADK Runner for the root_agent
runner = Runner(
    agent=loan_approval_workflow,
    session_service=session_service,
    app_name="national_bank_loan_workflow",
)
