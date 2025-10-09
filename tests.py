import asyncio

import dspy
from pydantic import BaseModel, Field

from src.agent_adapter import DSPyAdapter, PydanticAdapter
from src.agent_adapter.meta import AgentAdapter


async def test_adapters(*adapters: AgentAdapter):
    template = """
        What is the capital of Türkiye?
    """
    adapter1, adapter2 = adapters

    # class CapitalSignature(dspy.Signature):
    #     user_prompt: str = dspy.InputField()
    #     capital: str = dspy.OutputField(description="the capital")

    # class CapitalModel(BaseModel):
    #     capital: str = Field(..., description="the capital")

    # result1 = await adapter1.produce(template, output_type=CapitalSignature)
    result1 = await adapter1.produce(
        template,
        output_schema={"capital": str, "country": str},
        # string=True
        # whether to use string signatures instead runtime signature generation
        instructions="şehri tersten yaz",
    )
    # CapitalSignature works here too, but we don't need user_prompt field
    result2 = await adapter2.produce(
        template, output_schema={"capital": str, "country": str}
    )

    print("DSPy Result: ", result1)
    print("PyAI Result: ", result2)

    assert "arakna" in result1.capital.lower() and "ankara" in result2.capital.lower()


async def main():
    adapter1 = DSPyAdapter("groq/qwen/qwen3-32b")
    adapter2 = PydanticAdapter("groq:qwen/qwen3-32b")
    await test_adapters(adapter1, adapter2)


if __name__ == "__main__":
    asyncio.run(main())
