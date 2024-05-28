import asyncio
import os
from dotenv import load_dotenv
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel import Kernel
import math
from typing import Annotated

from semantic_kernel.functions.kernel_function_decorator import kernel_function

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


class Math:
    """
    Description: MathPlugin provides a set of functions to make Math calculations.

    Usage:
        kernel.add_plugin(MathPlugin(), plugin_name="math")

    Examples:
        {{math.Add}} => Returns the sum of input and amount (provided in the KernelArguments)
        {{math.Subtract}} => Returns the difference of input and amount (provided in the KernelArguments)
        {{math.Multiply}} => Returns the multiplication of input and number2 (provided in the KernelArguments)
        {{math.Divide}} => Returns the division of input and number2 (provided in the KernelArguments)
    """

    @kernel_function(
        description="Divide two numbers.",
        name="Divide",
    )
    def divide(
        self,
        number1: Annotated[float, "the first number to divide from"],
        number2: Annotated[float, "the second number to by"],
    ) -> Annotated[float, "The output is a float"]:
        return float(number1) / float(number2)

    @kernel_function(
        description="Multiply two numbers. When increasing by a percentage, don't forget to add 1 to the percentage.",
        name="Multiply",
    )
    def multiply(
        self,
        number1: Annotated[float, "the first number to multiply"],
        number2: Annotated[float, "the second number to multiply"],
    ) -> Annotated[float, "The output is a float"]:
        return float(number1) * float(number2)

    @kernel_function(
        description="Takes the square root of a number",
        name="Sqrt",
    )
    def square_root(
        self,
        number1: Annotated[float, "the number to take the square root of"],
    ) -> Annotated[float, "The output is a float"]:
        return math.sqrt(float(number1))

    @kernel_function(name="Add")
    def add(
        self,
        number1: Annotated[float, "the first number to add"],
        number2: Annotated[float, "the second number to add"],
    ) -> Annotated[float, "the output is a float"]:
        return float(number1) + float(number2)

    @kernel_function(
        description="Subtracts value to a value",
        name="Subtract",
    )
    def subtract(
        self,
        number1: Annotated[float, "the first number"],
        number2: Annotated[float, "the number to subtract"],
    ) -> Annotated[float, "the output is a float"]:
        return float(number1) - float(number2)


math_plugin = kernel.add_plugin(Math(), "Math")

sqrt_function = math_plugin["Sqrt"]

async def square_root():
    return await kernel.invoke(sqrt_function, number1=4)

print(asyncio.run(square_root()))