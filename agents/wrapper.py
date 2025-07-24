from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.language_models import LanguageModelInput, LanguageModelOutput
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from typing import List, Union, Any, Optional, Dict


def message_to_str(m: Any) -> str:
    if isinstance(m, dict):
        role = m.get("role", "user")
        content = m.get("content", "")
        sender = "Human" if role in ("user", "human") else "AI"
    elif isinstance(m, BaseMessage):
        sender = "Human" if m.type in ("human", "user") else "AI"
        content = m.content
    else:
        sender = "AI"
        content = str(m)
    return f"{sender}: {content}"


class ToolCallingLLMWrapper(Runnable[LanguageModelInput, LanguageModelOutput]):
    """
    Wrapper biến HuggingFacePipeline thành một đối tượng
    tương thích với create_react_agent (có bind_tools, tool_calls, v.v.)
    """

    def __init__(self, llm, tools: List[BaseTool]):
        self.llm = llm
        self.tools = {tool.name: tool for tool in (tools or [])}

    def bind_tools(
        self,
        tools: List[Union[BaseTool, dict]],
        **kwargs: Any,
    ):
        """
        Gắn công cụ vào model — cần thiết để dùng với create_react_agent.
        Trả về một bản sao với tools được gán.
        """
        # Chuyển danh sách tools thành BaseTool nếu cần
        base_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                # Nếu là dict (OpenAI-style tool schema), bạn có thể parse thành BaseTool
                # Ở đây tạm bỏ qua và chỉ giữ tên
                name = tool.get("function", {}).get("name") or tool.get("name")
                if name:
                    # Tạo một placeholder tool
                    from langchain_core.tools import Tool
                    base_tool = Tool(
                        name=name,
                        description="Tool from schema",
                        func=lambda *args, **kwargs: "Tool executed",
                    )
                    base_tool.return_direct = False
                    base_tools.append(base_tool)
            elif isinstance(tool, BaseTool):
                base_tools.append(tool)

        # Tạo bản sao với tools mới
        return ToolCallingLLMWrapper(
            llm=self.llm,
            tools=base_tools
        )

    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> LanguageModelOutput:
        from langchain_core.prompt_values import PromptValue

        if isinstance(input, PromptValue):
            prompt_str = input.to_string()
        elif isinstance(input, str):
            prompt_str = input
        elif isinstance(input, list):
            prompt_str = "\n".join(message_to_str(m) for m in input)
        else:
            prompt_str = str(input)

        result_text = self.llm.invoke(prompt_str)
        tool_call = self._parse_tool_call(result_text)

        if tool_call:
            return AIMessage(content="", tool_calls=[tool_call])
        return AIMessage(content=result_text)

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> LanguageModelOutput:
        from langchain_core.prompt_values import PromptValue

        if isinstance(input, PromptValue):
            prompt_str = input.to_string()
        elif isinstance(input, str):
            prompt_str = input
        elif isinstance(input, list):
            prompt_str = "\n".join(message_to_str(m) for m in input)
        else:
            prompt_str = str(input)

        result_text = await self.llm.ainvoke(prompt_str)
        tool_call = self._parse_tool_call(result_text)

        if tool_call:
            return AIMessage(content="", tool_calls=[tool_call])
        return AIMessage(content=result_text)

    def _parse_tool_call(self, text: str):
        import json
        import re
        
        # More flexible pattern matching
        pattern = r"\{[\s\S]*\"name\"[\s\S]*\"arguments\"[\s\S]*\}"
        match = re.search(pattern, text)
        
        if not match:
            return None

        try:
            json_str = match.group(0).replace("'", '"')  # Fix quotes
            data = json.loads(json_str)
            name = data["name"]
            args = data.get("arguments", {})
            if name in self.tools:
                return {
                    "name": name,
                    "args": args,
                    "id": f"call_{abs(hash(text)) % 1000000:06d}",
                    "type": "tool_call"
                }
        except Exception as e:
            print(f"[DEBUG] Failed to parse tool call: {e}")
            return None
        return None

    @property
    def _llm_type(self) -> str:
        return "custom_tool_calling_wrapper"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "llm": str(self.llm),
            "tool_names": list(self.tools.keys())
        }