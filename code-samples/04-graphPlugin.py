from openai import AzureOpenAI
import os
from openai import AzureOpenAI
import dotenv
from dotenv import load_dotenv
import msal
import requests
from msal import PublicClientApplication
from azure.identity import DefaultAzureCredential
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel import Kernel
from typing import Annotated
import asyncio
from semantic_kernel.functions.kernel_function_decorator import kernel_function

load_dotenv()

class TokenManager:
    token: str = "" #data member to store the bearer access token
    
class GraphPlugin:
    """
    this class GraphPlugin acts as a native function to give the functionality for the LLM to base its answers upon
    context derived from the protected Microsoft Graph Web API.
    
    """
    
    @kernel_function(
        description="To list the calendar events of the user such as meetings etc.",
        name = "ListCalendarEvents"
    )
    def ListCalenderEvents(
        self,
        user_query: Annotated[str, "the query of the user"]
    ) -> Annotated[str, "the output is a string variable"]:
        
        print("ListCalendarEvents function called")
        print("fetching answer .........")
        
        url="https://graph.microsoft.com/v1.0/me/events?$select=subject,body,bodyPreview,organizer,attendees,start,end,location"
        
        headers = {
            "Authorization": f"Bearer {TokenManager.token}",
            "Content-Type": "application/json"
        }
        
        get_url_response = requests.get(url, headers=headers)
        responseString = get_url_response.json()
        
        systemMessage = f"You are a helpful AI assistant meant to assist the user by answering their queries related to knowing the calendar events in \
        the microsoft graph API. you will be presented with the user query that the user asked and a JSON response of the graph API. Extract \
        information from that JSON response based on the user query and present it to the user in a readable format."
        
        systemPrompt = f"The user query is: {user_query}. The JSON response from the graph API is: {responseString}. Extract information from the JSON response based on the user query and present it to the user in a readable format."
        
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-15-preview"
        )
        
        chatResponse = client.chat.completions.create(
            model = os.getenv("AZURE_OPENAI_CHAT_COMPLETION_MODEL"),
            messages = [
                {
                    "role": "system",
                    "content": systemMessage
                },
                {
                  "role": "user",
                  "content":systemPrompt
                }
            ]
        )
        
        return chatResponse.choices[0].message.content
    
async def main():
    user_input = input("enter the user query") #take the user queries like "List the calendar events", "what was the last meeting", etc.
    
    client_id = os.getenv("AZURE_OPENAI_CLIENT_ID") #fill in the client id in the .env file
    tenantId = os.getenv("AZURE_OPENAI_TENANT_ID") #fill in the tenant id in the .env file
   
    authority = f"https://login.microsoftonline.com/{tenantId}"
    scopes = ["User.Read", "Calendars.Read", "Calendars.ReadWrite"]
    
    app=msal.PublicClientApplication(
        client_id,
        authority=authority,
        client_credential=None
    )
    
    flow = app.initiate_device_flow(scopes=scopes)
    print(flow["message"])
    result = app.acquire_token_by_device_flow(flow)
    
    TokenManager.token = result["access_token"]
    
    kernel = Kernel()
    
    service_id = "default"
    kernel.add_service(
        AzureChatCompletion(service_id=service_id,
                            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                            deployment_name=os.getenv("AZURE_OPENAI_CHAT_COMPLETION_MODEL"),
                            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    )
    
    graphPlugin = kernel.add_plugin(GraphPlugin() , "Graphplugin")
    calenderFunction = graphPlugin["ListCalendarEvents"]
    finalResult = await kernel.invoke(calenderFunction, user_query=user_input) #invoke the function "ListCalendarEvents" with the user query
    
    print("-----------------")
    print(finalResult)
    
if (__name__=="__main__"):
    asyncio.run(main())