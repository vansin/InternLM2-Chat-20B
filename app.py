# 导入所需的库
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import streamlit as st

from modelscope import snapshot_download

def on_btn_click():
    del st.session_state.messages
# 在侧边栏中创建一个标题和一个链接
with st.sidebar:
    max_length = st.slider("Max Length", min_value=32, max_value=2048, value=2048)
    top_p = st.slider("Top P", 0.0, 1.0, 0.8, step=0.01)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.01)
    st.button("Clear Chat History", on_click=on_btn_click)
    system_prompt = st.text_input("System_Prompt", "你是一个人工智能助手")

# 创建一个标题和一个副标题
st.title("💬 InternLM2-Chat-20B-4bits")
st.caption("🚀 A streamlit chatbot powered by InternLM2")

# 定义模型路径

model_id = 'Shanghai_AI_Laboratory/internlm2-chat-20b'

mode_name_or_path = snapshot_download(model_id, revision='master')


# 定义一个函数，用于获取模型和tokenizer
@st.cache_resource
def get_model():
    # 从预训练的模型中获取tokenizer
    tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, trust_remote_code=True)
    # 从预训练的模型中获取模型，并设置模型参数
    model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, trust_remote_code=True, load_in_4bit=True, device_map="auto")
    model.eval()  
    return tokenizer, model

# 加载InternLM的model和tokenizer
tokenizer, model = get_model()

# 如果session_state中没有"messages"，则创建一个包含默认消息的列表
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 遍历session_state中的所有消息，并显示在聊天界面上
for msg in st.session_state.messages:
    st.chat_message("user").write(msg[0])
    st.chat_message("assistant").write(msg[1])

# 如果用户在聊天输入框中输入了内容，则执行以下操作
if prompt := st.chat_input():
    # 在聊天界面上显示用户的输入
    st.chat_message("user").write(prompt)
    # 构建输入     
    response, history = model.chat(tokenizer, prompt, meta_instruction=system_prompt, history=st.session_state.messages)
    # 将模型的输出添加到session_state中的messages列表中
    st.session_state.messages.append((prompt, response))
    # 在聊天界面上显示模型的输出
    st.chat_message("assistant").write(response)
