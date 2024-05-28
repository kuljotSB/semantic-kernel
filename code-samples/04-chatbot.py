from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import KernelArguments
import os
import asyncio
from semantic_kernel.contents import ChatHistory
from dotenv import load_dotenv 

kernel = Kernel()

load_dotenv()

service_id = "default"
kernel.add_service(
    AzureChatCompletion(service_id=service_id,
                        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                        deployment_name=os.getenv("AZURE_OPENAI_CHAT_COMPLETION_MODEL"),
                        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
)

chat_plugin = kernel.add_plugin(parent_directory="../plugins/prompt_templates/", plugin_name="chat")
chat_function = chat_plugin["AIChat"]

chat_history = ChatHistory()
chat_history.add_system_message("You are a helpful AI assistant")


async def chat_session():
    user_input = " "
    while(user_input != "exit"):
        
     user_input = input("type exit to end the chat session or your message to extend the conversation \n")
     if user_input == "exit":
        return
     else:
       response = await kernel.invoke(chat_function, KernelArguments(query=user_input, chat_history=chat_history))
       print(response , "\n")
       chat_history.add_assistant_message(str(response))
       
asyncio.run(chat_session())