"""
Multi-Role Summarization with Gemini API + LangChain
----------------------------------------------------
Generates two summaries for a research abstract:
  1. Scientist-style (technical, precise)
  2. News Reporter-style (accessible, engaging)
"""

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

def create_summarization_chain():
    """Create an LLMChain for dual-role summarization."""

    # Directly using your API key (⚠️ Not recommended for production)
    api_key = "AIzaSyAw6HwSznXUGr-g5HhqlQp00OXCgeV-RLU"

    # Initialize Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=api_key,
        temperature=0.9
    )

    # Define structured prompt
    prompt = PromptTemplate(
        input_variables=["abstract"],
        template=(
            "You are given a research abstract. Provide two summaries:\n"
            "1. Scientist Summary: (technical, precise, domain-specific)\n"
            "2. News Reporter Summary: (engaging, simple, audience-friendly)\n\n"
            "Abstract:\n\"\"\"\n{abstract}\n\"\"\""
        )
    )

    return LLMChain(llm=llm, prompt=prompt)


def main():
    abstract = (
        "Quantum computing is emerging as a field that uses principles of quantum mechanics "
        "to perform calculations much faster than classical computers. Researchers are exploring "
        "its applications in cryptography, optimization, and materials science."
    )

    chain = create_summarization_chain()
    response = chain.run({"abstract": abstract})

    print("\n===== Multi-Role Summarization via Gemini + LangChain =====")
    print(response)


if __name__ == "__main__":
    main()
