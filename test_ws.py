import asyncio
import websockets
import json


async def run_ws_smoke_test():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        await ws.send(json.dumps({"action": "reset", "task_name": "gamma_scalping"}))
        resp = json.loads(await ws.recv())
        print("Reset Response:", list(resp.keys()))

        await ws.send(
            json.dumps(
                {
                    "action": "step",
                    "payload": {
                        "selected_strike": 4,
                        "selected_maturity": 0,
                        "direction": "sell",
                        "quantity": 1.0,
                        "reasoning": "gamma scalp",
                    },
                }
            )
        )
        resp = json.loads(await ws.recv())
        print("Step Response Keys:", list(resp.keys()))
        print("Done!")


if __name__ == "__main__":
    asyncio.run(run_ws_smoke_test())
