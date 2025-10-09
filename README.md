# AI Agent Adapter Library

A unified adapter library that provides a consistent interface for working with both Pydantic AI and DSPy agents. This library simplifies the process of creating and invoking AI agents with structured outputs by abstracting away framework-specific implementation details.

## Features

- **Unified Interface**: Common API for both Pydantic AI and DSPy frameworks
- **Dynamic Schema Generation**: Automatically creates Pydantic models from dictionary schemas
- **Type Safety**: Full type hint support with generic types
- **Flexible Configuration**: Configurable model parameters, temperature, and token limits
- **String Signatures**: Support for both class-based and string-based DSPy signatures
- **Async Support**: Built with async/await for modern Python applications

## Installation

```bash
uv venv venv
uv init
uv sync
```

```bash
python3 -m virtualenv venv
python3 -m pip install -r requirements.txt
```

## Requirements

- Python 3.10+
- dspy
- pydantic >= 2.0
- pydantic-ai

## Usage

### Quick Start

```python
import asyncio
from agent_adapter import DSPyAdapter, PydanticAdapter

async def main():
    # Initialize adapters
    dspy_adapter = DSPyAdapter("groq/qwen/qwen3-32b")
    pydantic_adapter = PydanticAdapter("groq:qwen/qwen3-32b")
    
    # Define your prompt (no curly braces needed)
    template = "What is the capital of France?"
    
    # Define output schema
    output_schema = {
        "capital": str,
        "country": str
    }
    
    # Get structured output from both adapters
    dspy_result = await dspy_adapter.produce(
        template,
        output_schema=output_schema
    )
    
    pydantic_result = await pydantic_adapter.produce(
        template,
        output_schema=output_schema
    )
    
    print(f"DSPy Result: {dspy_result.capital}")
    print(f"Pydantic AI Result: {pydantic_result.capital}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Complete Example

```python
import asyncio
from agent_adapter import DSPyAdapter, PydanticAdapter
from agent_adapter.meta import AgentAdapter

async def test_adapters(*adapters: AgentAdapter):
    template = """
        What is the capital of Türkiye?
    """
    
    adapter1, adapter2 = adapters
    
    # DSPy adapter with custom instructions
    result1 = await adapter1.produce(
        template,
        output_schema={"capital": str, "country": str},
        instructions="şehri tersten yaz",  # Custom instructions
    )
    
    # Pydantic AI adapter
    result2 = await adapter2.produce(
        template, 
        output_schema={"capital": str, "country": str}
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
```

### DSPy Adapter Examples

#### Basic Usage

```python
from agent_adapter import DSPyAdapter

async def analyze_sentiment():
    adapter = DSPyAdapter(
        model="groq/qwen/qwen3-32b",
        max_tokens=4000,
        temperature=0.7
    )
    
    template = "Analyze the sentiment of this review: The product exceeded my expectations!"
    
    output_schema = {
        "sentiment": str,
        "confidence": float,
        "summary": str
    }
    
    result = await adapter.produce(
        template=template,
        output_schema=output_schema
    )
    
    print(f"Sentiment: {result.sentiment}")
    print(f"Confidence: {result.confidence}")
    print(f"Summary: {result.summary}")
```

#### Using String Signatures

```python
async def extract_info_with_string_sig():
    adapter = DSPyAdapter("groq/qwen/qwen3-32b")
    
    template = "Extract key information from: John Doe works at Acme Corp as a Software Engineer"
    
    result = await adapter.produce(
        template=template,
        output_schema={
            "name": str,
            "company": str,
            "position": str
        },
        string=True  # Use string-based signatures
    )
    
    print(f"Name: {result.name}")
    print(f"Company: {result.company}")
    print(f"Position: {result.position}")
```

#### Custom Instructions

```python
async def custom_formatting():
    adapter = DSPyAdapter("groq/qwen/qwen3-32b")
    
    template = "Summarize the main points of quantum computing"
    
    result = await adapter.produce(
        template=template,
        output_schema={"summary": str, "key_concepts": str},
        instructions="Keep the response under 100 words and use simple language"
    )
    
    print(result.summary)
```

#### Complex Schema Example

```python
async def detailed_analysis():
    adapter = DSPyAdapter("groq/qwen/qwen3-32b")
    
    template = """
        Analyze this business report:
        Q4 2024 showed 25% revenue growth with strong performance in EMEA region.
        Customer satisfaction increased to 4.5/5 stars.
    """
    
    output_schema = {
        "revenue_growth": float,
        "top_region": str,
        "satisfaction_score": float,
        "overall_assessment": str,
        "recommendations": str
    }
    
    result = await adapter.produce(
        template=template,
        output_schema=output_schema,
        instructions="Provide actionable insights"
    )
    
    print(f"Growth: {result.revenue_growth}%")
    print(f"Top Region: {result.top_region}")
    print(f"Assessment: {result.overall_assessment}")
```

### Pydantic AI Adapter Examples

#### Basic Usage

```python
from agent_adapter import PydanticAdapter

async def classify_text():
    adapter = PydanticAdapter("groq:qwen/qwen3-32b")
    
    template = "Classify this email: Meeting scheduled for tomorrow at 2 PM"
    
    output_schema = {
        "category": str,
        "urgency": str,
        "action_required": bool
    }
    
    result = await adapter.produce(
        template=template,
        output_schema=output_schema
    )
    
    print(f"Category: {result.category}")
    print(f"Urgency: {result.urgency}")
    print(f"Action Required: {result.action_required}")
```

#### Data Extraction

```python
async def extract_structured_data():
    adapter = PydanticAdapter("groq:qwen/qwen3-32b")
    
    template = """
        Parse this invoice:
        Invoice #12345
        Date: 2024-10-10
        Amount: $1,250.00
        Customer: Acme Corp
    """
    
    output_schema = {
        "invoice_number": str,
        "date": str,
        "amount": float,
        "customer": str
    }
    
    result = await adapter.produce(
        template=template,
        output_schema=output_schema
    )
    
    print(f"Invoice: {result.invoice_number}")
    print(f"Amount: ${result.amount}")
```

### Comparing Both Adapters

```python
async def compare_adapters():
    template = "What are the benefits of renewable energy?"
    
    output_schema = {
        "main_benefits": str,
        "environmental_impact": str,
        "economic_impact": str
    }
    
    # DSPy approach
    dspy_adapter = DSPyAdapter("groq/qwen/qwen3-32b")
    dspy_result = await dspy_adapter.produce(
        template=template,
        output_schema=output_schema,
        instructions="Be concise and factual"
    )
    
    # Pydantic AI approach
    pydantic_adapter = PydanticAdapter("groq:qwen/qwen3-32b")
    pydantic_result = await pydantic_adapter.produce(
        template=template,
        output_schema=output_schema
    )
    
    print("=== DSPy Result ===")
    print(dspy_result)
    
    print("\n=== Pydantic AI Result ===")
    print(pydantic_result)
```

## API Reference

### DSPyAdapter

#### Constructor

```python
DSPyAdapter(
    model: str,
    max_tokens: int = 4000,
    temperature: float = 1.0,
    provider: dspy.Provider | None = None,
    **kwargs
)
```

**Parameters:**
- `model` (str): The model identifier (e.g., "groq/qwen/qwen3-32b", "gpt-4")
- `max_tokens` (int): Maximum tokens for generation (default: 4000)
- `temperature` (float): Sampling temperature (default: 1.0)
- `provider` (dspy.Provider | None): Optional DSPy provider instance
- `**kwargs`: Additional arguments passed to the DSPy LM constructor

#### Methods

##### `type_builder`

```python
async def type_builder(
    output_schema: dict[str, type],
    **kwargs
) -> str | dspy.Signature
```

Creates a DSPy signature from the output schema.

**Parameters:**
- `output_schema` (dict[str, type]): Dictionary mapping field names to types
- `string` (bool): If True, returns string signature; otherwise returns class-based signature

**Returns:** Either a string signature or a DSPy Signature class

##### `produce`

```python
async def produce(
    template: str,
    output_schema: dict[str, type],
    **kwargs
) -> T
```

Generates structured output using DSPy.

**Parameters:**
- `template` (str): The prompt as a plain string (no variable placeholders)
- `output_schema` (dict[str, type]): Dictionary defining output structure
- `string` (bool): Use string-based signatures (default: False)
- `instructions` (str): Optional instructions for the model
- `**kwargs`: Additional configuration options

**Returns:** A DSPy prediction object with fields matching the output schema

### PydanticAdapter

Inherits from `Agent[AgentDepsT, OutputDataT]`.

#### Methods

##### `type_builder`

```python
def type_builder(output_schema: dict[str, type]) -> type[BaseModel]
```

Creates a Pydantic model class from the output schema.

**Parameters:**
- `output_schema` (dict[str, type]): Dictionary mapping field names to types

**Returns:** A dynamically created Pydantic BaseModel class

##### `produce`

```python
async def produce(
    template: str,
    output_schema: dict,
    **kwargs
)
```

Invokes the Pydantic AI agent.

**Parameters:**
- `template` (str): The prompt as a plain string
- `output_schema` (dict): Dictionary defining output structure
- `**kwargs`: Additional arguments passed to the agent

**Returns:** The structured output matching the schema

## Type System

Both adapters support Python's type system for schema definitions:

```python
output_schema = {
    "text": str,
    "count": int,
    "score": float,
    "items": list,
    "metadata": dict,
    "is_valid": bool
}
```

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributing

Contributions are welcome! Please ensure your code follows the existing style and includes appropriate type hints.

## Support

For issues, questions, or contributions, please open an issue on GitHub.