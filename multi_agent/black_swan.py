import numpy as np
from pydantic import BaseModel
from typing import List

class BlackSwanEvent(BaseModel):
    event_type: str
    severity: float
    headline: str
    spot_impact: float
    variance_impact: float
    pre_signal_steps: int
    trigger_step: int
    news_step: int

class BlackSwanGenerator:
    """Deterministic event generator seeded per episode."""
    
    def __init__(self, rng: np.random.RandomState, episode_length: int = 300):
        self.events = self._schedule_events(rng, episode_length)
    
    def _schedule_events(self, rng: np.random.RandomState, episode_length: int) -> List[BlackSwanEvent]:
        """Schedule max 2 events per 100 steps, well-spaced."""
        events = []
        # Divide episode into 100-step windows
        for window_start in range(0, episode_length, 100):
            window_end = min(window_start + 100, episode_length)
            # [B1 FIX] Skip windows too small to safely place events
            if window_end - window_start < 30:
                continue
            n_events = rng.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
            
            if n_events >= 1:
                # First event: within first 60% of window, clamped to boundary
                first_end = min(window_start + 60, window_end - 5)
                if window_start + 15 < first_end:
                    step1 = rng.randint(window_start + 15, first_end)
                    events.append(self._generate_event(rng, step1))
                else:
                    continue  # can't fit even one event
            if n_events >= 2:
                # Second event: within last 40%, min 25 steps from first
                second_start = max(window_start + 65, step1 + 25)
                second_end = window_end - 5
                if second_start < second_end:
                    step2 = rng.randint(second_start, second_end)
                    events.append(self._generate_event(rng, step2))
        return events
        
    def _generate_event(self, rng: np.random.RandomState, trigger_step: int) -> BlackSwanEvent:
        event_types = ["crash", "boom", "pandemic", "war", "fed_rate", "sector_crisis"]
        event_type = rng.choice(event_types)
        severity = rng.uniform(0.0, 1.0)
        pre_signal_steps = rng.randint(1, 6) # 1 to 5
        news_step = max(0, trigger_step - pre_signal_steps)
        
        headline_clarity = rng.choice(["explicit", "ambiguous"])
        
        if event_type == "crash":
            headline = "BREAKING: Major bank declares bankruptcy" if headline_clarity == "explicit" else "Major institution faces liquidity concerns"
            spot_impact = rng.uniform(0.70, 0.85)
            variance_impact = rng.uniform(2.5, 5.0)
        elif event_type == "boom":
            headline = "BREAKING: FDA approves revolutionary treatment" if headline_clarity == "explicit" else "Market rallied on positive biotech developments"
            spot_impact = rng.uniform(1.15, 1.35)
            variance_impact = rng.uniform(1.5, 3.0)
        elif event_type == "pandemic":
            headline = "WHO declares global health emergency" if headline_clarity == "explicit" else "Unusual virus spread observed globally"
            spot_impact = rng.uniform(0.75, 0.90)
            variance_impact = rng.uniform(3.0, 6.0)
        elif event_type == "war":
            headline = "Military conflict erupts in major oil region" if headline_clarity == "explicit" else "Geopolitical tensions rise significantly"
            spot_impact = rng.uniform(0.80, 0.90)
            variance_impact = rng.uniform(2.0, 4.0)
        elif event_type == "fed_rate":
            headline = "Fed announces emergency rate cut of 100bps" if headline_clarity == "explicit" else "Unexpected shift in monetary policy announced"
            spot_impact = rng.uniform(1.05, 1.20)
            variance_impact = rng.uniform(1.5, 2.5)
        elif event_type == "sector_crisis":
            headline = "Entire crypto exchange collapses overnight" if headline_clarity == "explicit" else "Systemic issues detected in tech sector"
            spot_impact = rng.uniform(0.85, 0.95)
            variance_impact = rng.uniform(2.0, 3.5)
        else:
            # fallback
            headline = "Unknown event"
            spot_impact = 1.0
            variance_impact = 1.0
            
        return BlackSwanEvent(
            event_type=event_type,
            severity=severity,
            headline=headline,
            spot_impact=spot_impact,
            variance_impact=variance_impact,
            pre_signal_steps=pre_signal_steps,
            trigger_step=trigger_step,
            news_step=news_step
        )
