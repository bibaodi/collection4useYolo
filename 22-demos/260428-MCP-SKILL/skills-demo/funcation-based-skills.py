# file: skills_simple.py
from typing import Callable, Dict, Any

class Skill:
    """A simple skill wrapper"""
    def __init__(self, name: str, description: str, function: Callable, parameters: dict):
        self.name = name
        self.description = description
        self.function = function
        self.parameters = parameters
    
    def to_openai_tool(self) -> dict:
        """Convert to OpenAI function calling format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
    
    def execute(self, **kwargs) -> Any:
        return self.function(**kwargs)

# Define skill functions
def search_files(query: str, directory: str = "/home") -> str:
    """Search for files matching query"""
    import subprocess
    try:
        result = subprocess.run(
            ["find", directory, "-name", f"*{query}*"],
            capture_output=True, text=True, timeout=10
        )
        files = result.stdout.strip().split('\n')[:10]
        return '\n'.join(files) if files[0] else "No files found"
    except Exception as e:
        return f"Error: {e}"

def read_file(filepath: str) -> str:
    """Read file contents"""
    try:
        with open(filepath, 'r') as f:
            return f.read()[:2000]  # Limit output
    except Exception as e:
        return f"Error: {e}"

# Create skills
skills = [
    Skill(
        name="search_files",
        description="Search for files by name pattern",
        function=search_files,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Filename pattern to search"},
                "directory": {"type": "string", "description": "Directory to search in"}
            },
            "required": ["query"]
        }
    ),
    Skill(
        name="read_file",
        description="Read contents of a file",
        function=read_file,
        parameters={
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to the file"}
            },
            "required": ["filepath"]
        }
    )
]

# Test skills directly
if __name__ == "__main__":
    print("=== Testing Skills Directly ===")
    for skill in skills:
        print(f"\nSkill: {skill.name}")
        print(f"Description: {skill.description}")
        print(f"OpenAI format: {skill.to_openai_tool()}")
    
    # Execute a skill
    print("\n=== Executing search_files skill ===")
    result = skills[0].execute(query=".py", directory="/home")
    print(result)

