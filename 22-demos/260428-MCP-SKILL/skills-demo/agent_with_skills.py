# file: agent_with_skills.py
import json
from openai import OpenAI
from skills_class import SkillRegistry, SystemInfoSkill, PackageManagerSkill

class SimpleAgent:
    """Simple agent using skills (OpenAI example)"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.registry = SkillRegistry()
        self._setup_skills()
    
    def _setup_skills(self):
        """Register available skills"""
        self.registry.register(SystemInfoSkill())
        self.registry.register(PackageManagerSkill())
    
    def run(self, user_message: str, max_iterations: int = 5) -> str:
        """Run agent with tool calling loop"""
        
        messages = [
            {"role": "system", "content": "You are a helpful Linux system assistant. Use tools when needed."},
            {"role": "user", "content": user_message}
        ]
        
        tools = self.registry.get_all_tool_definitions()
        
        for i in range(max_iterations):
            print(f"\n--- Iteration {i+1} ---")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools if tools else None
            )
            
            msg = response.choices[0].message
            
            # Add assistant response to history
            messages.append(msg.to_dict())
            
            # Check if agent wants to call a tool
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    
                    print(f"Calling skill: {func_name}")
                    print(f"Arguments: {func_args}")
                    
                    # Execute the skill
                    result = self.registry.execute(func_name, **func_args)
                    print(f"Result: {result[:200]}...")
                    
                    # Return result to LLM
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
            else:
                # No more tool calls - return final response
                return msg.content
        
        return "Max iterations reached"

# Usage (requires OPENAI_API_KEY)
if __name__ == "__main__":
    import os
    
    agent = SimpleAgent(api_key=os.environ.get("OPENAI_API_KEY"))
    
    response = agent.run("What's my current memory usage and check if nginx is available in apt?")
    print(f"\n=== Final Response ===\n{response}")

