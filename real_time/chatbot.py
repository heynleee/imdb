from transformers import pipeline, Conversation

chatbot = pipeline('conversational')

def chatbot_response(user_input):
    conversation = Conversation(user_input)
    result = chatbot([conversation])
    return conversation.generated_responses[0]
