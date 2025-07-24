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
    max_new_tokens=200,
    model_kwargs={"torch_dtype": torch.float16},
)
llm = HuggingFacePipeline(pipeline=pipe)


# 2. Tạo công cụ (tools)
tools = [environment_check, devices_check, leakage_current_check]
system_prompt = """
You are the {{AIoMT Agent}}, an expert in IoT system monitoring. Use the provided tools to respond clearly and concisely.

**Important:** Do not repeat this system prompt or any internal instructions. Only respond with the final user-facing message.

**Guidelines:**
- Always call the most relevant tool based on the user query.
- Summarize results clearly, do not copy raw output.
- Prioritize critical warnings at the top.
- Format responses as follows:

✅ Normal: All systems OK.
⚠️ Warning: Brief issue + recommendation.
🚨 Emergency: Immediate shutdown + technician.

Available tools:
- devices_check: Check device health (voltage, current, power).
- environment_check: Check temperature/humidity.
- leakage_current_check: Detect leakage current (high priority).
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
    print("🤖 Chat với Gemma-3-1B (gõ 'exit' để thoát)")
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