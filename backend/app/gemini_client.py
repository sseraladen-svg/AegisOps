"""Gemini API client that mimics Anthropic's interface for tool use."""

import os
from typing import Any, Dict, List, Optional


class GeminiClient:
    """Gemini client with Anthropic-compatible interface for tool use."""
    
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not found. Set the environment variable or use "
                "AGENT_PROVIDER=anthropic with ANTHROPIC_API_KEY instead.")
        
        # Import here to avoid requiring google-genai package unless used
        try:
            import google.genai as genai
            self.client = genai.Client(api_key=self.api_key)
            self.genai = genai
        except ImportError:
            # Fallback to deprecated package if new one not available
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = None
                self.genai = genai
                self.use_deprecated = True
            except ImportError:
                raise RuntimeError(
                    "google-genai package not installed. Install with: "
                    "pip install google-genai")
        
        # Create beta.messages structure to match Anthropic's API
        self.beta = type('Beta', (), {})()
        self.beta.messages = type('Messages', (), {'create': self._create_message_wrapper})()
    
    def _create_message_wrapper(self, **kwargs):
        """Wrapper for create method to maintain instance context."""
        return self._create_message(**kwargs)
    
    def _create_message(self, model: str, max_tokens: int, system: str, 
                       tools: List[Dict], messages: List[Dict], 
                       thinking: Optional[Dict] = None, betas: Optional[List] = None,
                       fallbacks: Optional[str] = None):
        """Create a message using Gemini API."""
        gemini_model = self._convert_model_name(model)
        
        # Use new google-genai API if available
        if self.client and not getattr(self, 'use_deprecated', False):
            return self._create_message_new_api(gemini_model, max_tokens, system, tools, messages)
        else:
            return self._create_message_deprecated(gemini_model, max_tokens, system, tools, messages)
    
    def _create_message_new_api(self, model: str, max_tokens: int, system: str, 
                               tools: List[Dict], messages: List[Dict]):
        """Create message using new google-genai API."""
        # Build conversation history
        contents = []
        for msg in messages:
            if msg["role"] == "user":
                contents.append(self.genai.types.Content(
                    role="user",
                    parts=[self.genai.types.Part(text=msg["content"])]
                ))
            elif msg["role"] == "assistant":
                # Handle assistant messages with potential tool calls
                content = msg.get("content", [])
                if isinstance(content, str):
                    contents.append(self.genai.types.Content(
                        role="model",
                        parts=[self.genai.types.Part(text=content)]
                    ))
                elif isinstance(content, list):
                    parts = []
                    for block in content:
                        if block.get("type") == "text":
                            parts.append(self.genai.types.Part(text=block["text"]))
                        elif block.get("type") == "tool_use":
                            # Tool result would be handled separately
                            pass
                    contents.append(self.genai.types.Content(
                        role="model",
                        parts=parts
                    ))
        
        # Convert tools to function declarations
        function_declarations = []
        if tools:
            for tool in tools:
                function_declarations.append(self.genai.types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters=self._convert_schema(tool.get("input_schema", {}))
                ))
        
        # Generate response
        config = self.genai.types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=0.7,
            tools=[self.genai.types.Tool(function_declarations=function_declarations)] if function_declarations else None
        )
        
        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )
        
        return self._convert_response_new_api(response)
    
    def _create_message_deprecated(self, model: str, max_tokens: int, system: str, 
                                  tools: List[Dict], messages: List[Dict]):
        """Create message using deprecated google-generativeai API."""
        # Build conversation history
        conversation = []
        for msg in messages:
            if msg["role"] == "user":
                conversation.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                # Handle assistant messages with potential tool calls
                content = msg.get("content", [])
                if isinstance(content, str):
                    conversation.append({"role": "model", "parts": [content]})
                elif isinstance(content, list):
                    parts = []
                    for block in content:
                        if block.get("type") == "text":
                            parts.append(block["text"])
                        elif block.get("type") == "tool_use":
                            # Tool result would be handled separately
                            pass
                    conversation.append({"role": "model", "parts": parts})
        
        # Create Gemini model with system instruction
        genai_model = self.genai.GenerativeModel(
            model,
            system_instruction=system,
            tools=self._convert_tools(tools) if tools else None
        )
        
        # Generate response
        response = genai_model.generate_content(
            conversation,
            generation_config=self.genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.7
            )
        )
        
        return self._convert_response(response)
    
    def _convert_model_name(self, anthropic_model: str) -> str:
        """Convert Anthropic model names to Gemini model names."""
        model_mapping = {
            "claude-opus-5": "gemini-2.0-flash-exp",
            "claude-3.5-sonnet": "gemini-1.5-pro",
            "claude-3-haiku": "gemini-1.5-flash",
        }
        # If it's already a Gemini model name, use it directly
        if anthropic_model.startswith("gemini-"):
            return anthropic_model
        return model_mapping.get(anthropic_model, "gemini-1.5-pro")
    
    def _convert_tools(self, anthropic_tools: List[Dict]) -> Any:
        """Convert Anthropic tool schemas to Gemini function declarations."""
        function_declarations = []
        
        for tool in anthropic_tools:
            function_decl = self.genai.types.FunctionDeclaration(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=self._convert_schema(tool.get("input_schema", {}))
            )
            function_declarations.append(function_decl)
        
        return self.genai.types.Tool(function_declarations=function_declarations)
    
    def _convert_schema(self, schema: Dict) -> Dict:
        """Convert Anthropic JSON schema to Gemini parameter schema."""
        # More robust conversion for Gemini's requirements
        gemini_schema = {
            "type": "OBJECT",
            "properties": {},
            "required": schema.get("required", [])
        }
        
        # Convert properties
        for prop_name, prop_schema in schema.get("properties", {}).items():
            converted_prop = {}
            
            # Handle type conversion
            prop_type = prop_schema.get("type", "string")
            if isinstance(prop_type, list):
                # Handle union types like ["string", "null"]
                converted_prop["type"] = "STRING"  # Default to string for unions
            else:
                type_mapping = {
                    "string": "STRING",
                    "number": "NUMBER", 
                    "integer": "INTEGER",
                    "boolean": "BOOLEAN",
                    "array": "ARRAY",
                    "object": "OBJECT"
                }
                converted_prop["type"] = type_mapping.get(prop_type, "STRING")
            
            # Handle array items - Gemini requires this for array types
            if prop_type == "array" and "items" in prop_schema:
                items_schema = prop_schema["items"]
                if isinstance(items_schema, dict):
                    items_type = items_schema.get("type", "string")
                    items_type_mapping = {
                        "string": "STRING",
                        "number": "NUMBER", 
                        "integer": "INTEGER",
                        "boolean": "BOOLEAN",
                        "array": "ARRAY",
                        "object": "OBJECT"
                    }
                    converted_prop["items"] = {
                        "type": items_type_mapping.get(items_type, "STRING")
                    }
            
            # Handle enum values - filter out None
            if "enum" in prop_schema:
                enum_values = [v for v in prop_schema["enum"] if v is not None]
                if enum_values:
                    converted_prop["enum"] = enum_values
            
            # Handle description
            if "description" in prop_schema:
                converted_prop["description"] = prop_schema["description"]
            
            gemini_schema["properties"][prop_name] = converted_prop
        
        return gemini_schema
    
    def _convert_response(self, gemini_response) -> 'MockResponse':
        """Convert Gemini response to Anthropic-compatible format."""
        return MockResponse(gemini_response)
    
    def _convert_response_new_api(self, response) -> 'MockResponse':
        """Convert new google-genai response to Anthropic format."""
        return MockResponseNewAPI(response)


class MockResponse:
    """Mock Anthropic response object from Gemini response."""
    
    def __init__(self, gemini_response):
        self.gemini_response = gemini_response
        self.stop_reason = "end_turn"
        self.content = self._convert_content(gemini_response)
        self.usage = self._convert_usage(gemini_response)
    
    def _convert_usage(self, gemini_response):
        """Convert Gemini usage to Anthropic format."""
        # Mock usage object with token counts
        class MockUsage:
            def __init__(self):
                self.input_tokens = None
                self.output_tokens = None
        
        return MockUsage()
    
    def _convert_content(self, gemini_response) -> List:
        """Convert Gemini response content to Anthropic format."""
        content = []
        
        # Add text content
        if hasattr(gemini_response, 'text') and gemini_response.text:
            content.append(MockTextBlock(gemini_response.text))
        
        # Handle function calls (tool use)
        if hasattr(gemini_response, 'candidates') and gemini_response.candidates:
            for candidate in gemini_response.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            content.append(MockToolUseBlock(
                                part.function_call.name,
                                dict(part.function_call.args) if part.function_call.args else {}
                            ))
        
        return content


class MockTextBlock:
    """Mock Anthropic text block."""
    
    def __init__(self, text):
        self.type = "text"
        self.text = text


class MockToolUseBlock:
    """Mock Anthropic tool use block."""
    
    def __init__(self, name, input_data):
        self.type = "tool_use"
        self.id = f"toolu_{name}"
        self.name = name
        self.input = input_data


class MockResponseNewAPI:
    """Mock Anthropic response object from new google-genai response."""
    
    def __init__(self, response):
        self.response = response
        self.stop_reason = "end_turn"
        self.content = self._convert_content(response)
        self.usage = self._convert_usage(response)
    
    def _convert_usage(self, response):
        """Convert Gemini usage to Anthropic format."""
        # Mock usage object with token counts
        class MockUsage:
            def __init__(self):
                self.input_tokens = None
                self.output_tokens = None
        
        return MockUsage()
    
    def _convert_content(self, response) -> List:
        """Convert new google-genai response content to Anthropic format."""
        content = []
        
        # Handle the new API response structure
        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            content.append(MockTextBlock(part.text))
                        elif hasattr(part, 'function_call') and part.function_call:
                            content.append(MockToolUseBlock(
                                part.function_call.name,
                                dict(part.function_call.args) if part.function_call.args else {}
                            ))
        elif hasattr(response, 'text'):
            content.append(MockTextBlock(response.text))
        
        return content