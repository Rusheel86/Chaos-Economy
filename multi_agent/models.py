from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class AgentRole(str, Enum):
    TRADER = "trader"
    MARKET_MAKER = "market_maker"
    OVERSIGHT = "oversight"

class AgentState(BaseModel):
    """Per-agent state tracking."""
    agent_id: str
    role: AgentRole
    cash_balance: float = 100_000.0
    positions: List[Dict] = Field(default_factory=list)
    portfolio_pnl: float = 0.0
    portfolio_delta: float = 0.0
    portfolio_gamma: float = 0.0
    portfolio_vega: float = 0.0
    fines_received: float = 0.0
    is_halted: bool = False

class MarketMakerAction(BaseModel):
    """MM sets bid-ask spreads."""
    atm_spread: float = Field(0.02, ge=0.001, le=0.20)
    otm_spread: float = Field(0.04, ge=0.001, le=0.30)
    itm_spread: float = Field(0.03, ge=0.001, le=0.25)
    skew_adjustment: float = Field(0.0, ge=-0.05, le=0.05)
    reasoning: str = ""

class OversightAction(BaseModel):
    """Oversight flags manipulation."""
    flagged_agents: List[str] = Field(default_factory=list)
    flag_type: str = "none"  # "spoofing" | "wash_trading" | "gamma_squeeze" | "none"
    fine_amount: float = 0.0
    halt_strikes: List[int] = Field(default_factory=list)
    reasoning: str = ""

class MultiAgentObservation(BaseModel):
    """Observation tailored to each agent's role."""
    agent_id: str
    role: AgentRole
    iv_surface: List[List[float]]
    spot_price: float
    mm_spreads: Dict[str, float]
    own_greeks: Dict[str, float]
    own_pnl: float
    own_positions: List[Dict]
    own_cash: float
    step_number: int
    steps_remaining: int
    # Oversight-only
    all_agent_pnls: Optional[Dict[str, float]] = None
    trade_log: Optional[List[Dict]] = None
