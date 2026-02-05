import asyncio
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from app.tools import evaluate_by_url, evaluate_manual

llm = ChatOllama(model="deepseek-v3.1:671b-cloud", base_url="http://localhost:11434", temperature=0)

agent = create_agent(
    model=llm,
    tools=[evaluate_by_url, evaluate_manual],
    system_prompt="""You are a technical evaluation bot.
- STRICT RULE: Do not ask follow-up questions. 
- If you see a District name and numbers like '21.26 m2' or '21/22 floor', IMMEDIATELY call 'evaluate_manual'.
- Use defaults for missing fields: ceiling_height=2.7, finishing='Без отделки', rooms=1.
- If the tool 'evaluate_by_url' returns an error, DO NOT give up. Immediately try to find parameters in the user's previous messages and call 'evaluate_manual'.
- Your final response must be in RUSSIAN and contain the PRICE.
- Never explain your inability to calculate if parameters are present in history.
"""
)


async def main():
    print("AI Real Estate Agent Started. Type 'q' to exit.")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['q', 'exit']: break

        async for event in agent.astream({"messages": [HumanMessage(content=user_input)]}, stream_mode="values"):
            last_msg = event["messages"][-1]
            if last_msg.type == "ai" and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    print(f"--- [CALLING TOOL: {tc['name']}] ---")

        if last_msg.content:
            print(f"\nModel: {last_msg.content}\n")


if __name__ == "__main__":
    asyncio.run(main())