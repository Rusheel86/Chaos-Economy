from typing import List, Dict, Optional, Any
import numpy as np

class NewsListing:
    def __init__(self, listing_id: str, seller_id: str, price: float, content: str, is_genuine: bool, target_buyers: List[str], posted_step: int = 0):
        self.listing_id = listing_id
        self.seller_id = seller_id
        self.price = price
        self.content = content
        self.is_genuine = is_genuine
        self.target_buyers = target_buyers
        self.posted_step = posted_step  # [B2] For expiry tracking

class NewsMarketplace:
    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
        self.listings: List[NewsListing] = []
        self.transaction_log: List[Dict[str, Any]] = []
        self.next_listing_id = 0
    
    def post_listing(self, seller_id: str, price: float, content: str, target: str = "all", current_step: int = 0) -> Optional[NewsListing]:
        """Agent posts intel for sale. 80% chance it's genuine."""
        # [M4 FIX] Reject empty/trivial content
        if not content or len(content.strip()) < 10:
            return None
        
        is_genuine = self.rng.random() < 0.8
        target_buyers = [target] if target != "all" else ["all"]
        listing = NewsListing(
            listing_id=f"intel_{self.next_listing_id}",
            seller_id=seller_id,
            price=price,
            content=content,
            is_genuine=is_genuine,
            target_buyers=target_buyers,
            posted_step=current_step
        )
        self.listings.append(listing)
        self.next_listing_id += 1
        return listing
    
    def buy_intel(self, buyer_id: str, listing_id: str, agent_states: Dict[str, Any], step: int) -> Optional[str]:
        """Buyer pays cash, receives content."""
        # [B3 FIX] Prevent re-purchasing the same listing
        if any(t["buyer_id"] == buyer_id and t["listing_id"] == listing_id
               for t in self.transaction_log):
            return None
        
        listing = next((l for l in self.listings if l.listing_id == listing_id), None)
        if not listing:
            return None
            
        buyer_state = agent_states.get(buyer_id)
        seller_state = agent_states.get(listing.seller_id)
        
        if buyer_state and seller_state and buyer_state.cash_balance >= listing.price:
            buyer_state.cash_balance -= listing.price
            seller_state.cash_balance += listing.price
            
            # [H2 FIX] Penalize seller for selling fake intel
            if not listing.is_genuine and seller_state:
                seller_state.cash_balance -= listing.price * 0.5  # 50% clawback
            
            self.transaction_log.append({
                "step": step,
                "buyer_id": buyer_id,
                "seller_id": listing.seller_id,
                "listing_id": listing.listing_id,
                "price": listing.price,
                "is_genuine": listing.is_genuine,
                "content": listing.content
            })
            return listing.content
        return None
    
    def get_available_listings(self, agent_id: str, current_step: int = 0, max_age: int = 15) -> List[Dict[str, Any]]:
        """What's for sale that this agent can see."""
        available = []
        for l in self.listings:
            if l.seller_id == agent_id:
                continue
            # [B2 FIX] Skip expired listings
            if current_step > 0 and current_step - l.posted_step > max_age:
                continue
            # Skip already-purchased listings for this agent
            if any(t["buyer_id"] == agent_id and t["listing_id"] == l.listing_id
                   for t in self.transaction_log):
                continue
            if "all" in l.target_buyers or agent_id in l.target_buyers:
                available.append({
                    "listing_id": l.listing_id,
                    "seller_id": l.seller_id,
                    "price": l.price
                })
        return available
