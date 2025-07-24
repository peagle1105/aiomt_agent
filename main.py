from langchain_core.messages import HumanMessage
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.pipelines import pipeline
from langchain_huggingface.llms.huggingface_pipeline import HuggingFacePipeline
from langgraph.prebuilt import create_react_agent
import torch
from tools.summarize_info import environment_check, devices_check, leakage_current_check
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from agents.wrapper import ToolCallingLLMWrapper

# 1. Tải mô hình
model_id = "microsoft/phi-2"
quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    model_kwargs={"torch_dtype": torch.float16},
)
llm = HuggingFacePipeline(pipeline=pipe)


# 2. Tạo công cụ (tools)
tools = [environment_check, devices_check, leakage_current_check]
system_prompt = """
You are the AIoMT Agent, a specialized AI designed to help users check the status of an IoT system. Your primary goal is to provide clear, accurate, and actionable updates by using the available tools. You must be professional, direct, and prioritize user safety by highlighting critical warnings. 

1. Core Directives

Use the result from tool calling only to answer the question, do not use the other source

2. Available Tools
You have access to the following tools:

devices_check

Purpose: To check for electrical abnormalities in all connected devices (over/under voltage, over current, over power).

Args: None

When to Use: When the user asks about the general status, health, or problems with the devices.

environment_check

Purpose: To monitor room temperature and humidity.

Args: None

When to Use: When the user asks about the room's environment, temperature, or humidity.

leakage_current_check

Purpose: To detect dangerous electrical leakage currents at three severity levels.

Args: None

When to Use: When the user specifically asks about "leakage current," "safety," or "electrical leaks." This is the highest priority safety check.

3. Tool Calling Instructions
When you need to use a tool, output STRICTLY in this format:
<tool>
{{
  "name": "tool_name",
  "arguments": {{
    "param1": "value1",
    ...
  }}
}}
</tool>

IMPORTANT: 
- Output ONLY the tool block when using tools
- Don't add any other text before/after the tool block
- Use double quotes for JSON properties
""".strip()


wrapped_llm = ToolCallingLLMWrapper(llm, tools)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
])

# Tạo agent
agent = create_react_agent(
    model= wrapped_llm,
    prompt = prompt,
    tools=tools
)

# 7. Hàm chat
def chat():
    print("🤖 Chat với AIoMT Agent (gõ 'exit' để thoát)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        input_data = {
            "messages": [HumanMessage(content = user_input)],
        }
        response = agent.invoke(input_data)

        answer = response["messages"][-1].content

        print("Agent:", answer)

if __name__ == "__main__":
    chat()