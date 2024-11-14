closed_book_system = """Given a claim, classify the claim based on your parametric knowledge.

Use the following format to provide your answer:
Prediction: [SUPPORT or REFUTE]
Confidence Level: [please show the percentage]

The confidence level indicates the degree of certainty you have about your answer and is represented as a percentage. 
For instance, if your confidence level is 80%, it means you are 80% certain that your answer is correct and there is a 20% chance that it may be incorrect."""

open_book_system = """Given a claim and evidence  (which can be text, table, or an image), determine whether the claim is SUPPORT or REFUTE by the evidence. 
            
Use the following format to provide your answer:
Prediction: [SUPPORT or REFUTE]
Explanation: [put your evidence and step-by-step reasoning here]
Confidence Level: [please show the percentage]

The confidence level indicates the degree of certainty you have about your answer and is represented as a percentage. 
For instance, if your confidence level is 80%, it means you are 80% certain that your answer is correct and there is a 20% chance that it may be incorrect."""

chain_of_thought = """Given a claim, classify the claim based on your parametric knowledge.

Chain of Thought:
Before providing your prediction, explain your reasoning step-by-step based on the knowledge you have about the claim. Break down any historical, factual, or contextual information that leads to your final classification.

Use the following format to provide your answer:
- Chain of Thought: [Step-by-step reasoning here]
- Prediction: [SUPPORT or REFUTE]
- Confidence Level: [show the percentage]

The confidence level indicates the degree of certainty you have about your answer and is represented as a percentage. For instance, if your confidence level is 80%, it means you are 80% certain that your answer is correct and there is a 20% chance that it may be incorrect.
"""