from typing import List, Dict, Any

class MessageChannel:
    """Supports DM (1:1), group chat, and broadcast."""
    
    CHANNEL_TYPES = ["dm", "group", "broadcast"]
    
    def __init__(self):
        self.channels: Dict[str, List[str]] = {}  # channel_id -> member list
        self.message_log: List[Dict[str, Any]] = []  # ALL messages (oversight can see)
        self.next_group_id = 0
    
    def send_dm(self, sender: str, recipient: str, message: str, current_step: int) -> None:
        """Private message. Only sender + recipient see it.
        BUT: oversight can subpoena the message log."""
        self.message_log.append({
            "type": "dm", "sender": sender, "recipient": recipient,
            "message": message, "step": current_step
        })
    
    def create_group(self, creator: str, members: List[str]) -> str:
        """Create a group chat. Members can coordinate."""
        channel_id = f"group_{self.next_group_id}"
        self.next_group_id += 1
        # ensure creator is in members
        if creator not in members:
            members.append(creator)
        self.channels[channel_id] = members
        return channel_id
    
    def send_group(self, sender: str, channel_id: str, message: str, current_step: int) -> None:
        """Send to group. All members see it."""
        if channel_id in self.channels and sender in self.channels[channel_id]:
            self.message_log.append({
                "type": "group", "sender": sender, "recipient": channel_id,
                "message": message, "step": current_step
            })
    
    def broadcast(self, sender: str, message: str, current_step: int) -> None:
        """Public broadcast. Everyone sees it including oversight."""
        self.message_log.append({
            "type": "broadcast", "sender": sender, "recipient": "all",
            "message": message, "step": current_step
        })

    def get_inbox(self, agent_id: str, current_step: int) -> List[Dict[str, Any]]:
        """Get messages for this agent at the current step."""
        inbox = []
        for msg in self.message_log:
            if msg["step"] != current_step:
                continue
            if msg["type"] == "dm" and msg["recipient"] == agent_id:
                inbox.append(msg)
            elif msg["type"] == "group":
                channel_members = self.channels.get(msg["recipient"], [])
                if agent_id in channel_members and msg["sender"] != agent_id:
                    inbox.append(msg)
            elif msg["type"] == "broadcast" and msg["sender"] != agent_id:
                inbox.append(msg)
        return inbox
