#!/usr/bin/env python3
"""Quick test of gpt-5 Responses API."""

import os
from openai import OpenAI
import config

def test_gpt5():
    """Test gpt-5 with responses.create()"""

    client = OpenAI(api_key=config.OPENAI_API_KEY)

    print(f"Testing model: {config.OPENAI_MODEL}")
    print(f"API Key present: {bool(config.OPENAI_API_KEY)}")
    print()

    # Test 1: Simple text generation
    print("=" * 80)
    print("TEST 1: Simple text generation")
    print("=" * 80)
    try:
        response = client.responses.create(
            model=config.OPENAI_MODEL,
            input="Say 'Hello, I am working!' and nothing else."
        )
        output = response.output_text
        print(f"✓ Response received: {len(output) if output else 0} chars")
        print(f"Output: {output}")
        print()
    except Exception as e:
        print(f"✗ Error: {e}")
        print()

    # Test 2: JSON generation (like report_generator)
    print("=" * 80)
    print("TEST 2: JSON generation (like report_generator)")
    print("=" * 80)

    system_prompt = """You are an expert scientific report generator.
Generate a JSON response with this structure:
{
    "answer": "Your answer here",
    "is_complete": true
}
"""

    user_prompt = """User Question: What is calcium imaging?

Please provide a brief answer in JSON format."""

    combined_input = f"{system_prompt}\n\n{user_prompt}"

    try:
        print(f"Input length: {len(combined_input)} chars")
        response = client.responses.create(
            model=config.OPENAI_MODEL,
            input=combined_input,
            max_output_tokens=500
        )
        output = response.output_text
        print(f"✓ Response received: {len(output) if output else 0} chars")
        print(f"Output: {output}")
        print()
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        print()

if __name__ == "__main__":
    test_gpt5()
