# file: skills_class.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseSkill(ABC):
    """Abstract base class for skills"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Skill name"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Skill description for agent"""
        pass
    
    @property
    def parameters_schema(self) -> dict:
        """JSON Schema for parameters"""
        return {"type": "object", "properties": {}}
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the skill"""
        pass
    
    def to_tool_definition(self) -> dict:
        """Export as tool definition"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema
            }
        }

class SystemInfoSkill(BaseSkill):
    """Get system information"""
    
    @property
    def name(self) -> str:
        return "get_system_info"
    
    @property
    def description(self) -> str:
        return "Get Linux system information (CPU, memory, disk)"
    
    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "info_type": {
                    "type": "string",
                    "enum": ["cpu", "memory", "disk", "all"],
                    "description": "Type of system info to retrieve"
                }
            },
            "required": ["info_type"]
        }
    
    def execute(self, info_type: str = "all") -> str:
        import subprocess
        
        commands = {
            "cpu": "lscpu | grep -E 'Model name|CPU\(s\)|Architecture'",
            "memory": "free -h",
            "disk": "df -h /",
            "all": "echo '=== CPU ===' && lscpu | grep -E 'Model name|CPU\\(s\\)' && echo '=== MEMORY ===' && free -h && echo '=== DISK ===' && df -h /"
        }
        
        try:
            result = subprocess.run(commands[info_type], shell=True, capture_output=True, text=True)
            return result.stdout or "No output"
        except Exception as e:
            return f"Error: {e}"

class PackageManagerSkill(BaseSkill):
    """Debian package management"""
    
    @property
    def name(self) -> str:
        return "manage_packages"
    
    @property
    def description(self) -> str:
        return "Search, check, or list installed packages on Debian"
    
    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "info", "list_installed"],
                    "description": "Action to perform"
                },
                "package_name": {
                    "type": "string",
                    "description": "Package name (for search/info)"
                }
            },
            "required": ["action"]
        }
    
    def execute(self, action: str, package_name: str = "") -> str:
        import subprocess
        
        commands = {
            "search": f"apt-cache search {package_name} | head -20",
            "info": f"apt-cache show {package_name} | head -30",
            "list_installed": "dpkg -l | tail -n +6 | head -20"
        }
        
        try:
            result = subprocess.run(commands[action], shell=True, capture_output=True, text=True)
            return result.stdout or "No results"
        except Exception as e:
            return f"Error: {e}"

# Skill registry
class SkillRegistry:
    """Registry to manage skills"""
    
    def __init__(self):
        self._skills: Dict[str, BaseSkill] = {}
    
    def register(self, skill: BaseSkill):
        self._skills[skill.name] = skill
        print(f"Registered skill: {skill.name}")
    
    def get(self, name: str) -> BaseSkill:
        return self._skills.get(name)
    
    def list_skills(self) -> list:
        return list(self._skills.keys())
    
    def get_all_tool_definitions(self) -> list:
        return [s.to_tool_definition() for s in self._skills.values()]
    
    def execute(self, name: str, **kwargs) -> str:
        skill = self._skills.get(name)
        if not skill:
            return f"Error: Skill '{name}' not found"
        return skill.execute(**kwargs)

# Usage
if __name__ == "__main__":
    registry = SkillRegistry()
    
    # Register skills
    registry.register(SystemInfoSkill())
    registry.register(PackageManagerSkill())
    
    print("\n=== All Skills ===")
    print(registry.list_skills())
    
    print("\n=== Tool Definitions (for LLM) ===")
    import json
    print(json.dumps(registry.get_all_tool_definitions(), indent=2))
    
    print("\n=== Execute SystemInfoSkill ===")
    print(registry.execute("get_system_info", info_type="memory"))
    
    print("\n=== Execute PackageManagerSkill ===")
    print(registry.execute("manage_packages", action="search", package_name="python"))

