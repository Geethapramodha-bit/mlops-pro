
def get_smalltalk_response(question: str) -> str | None:
    normalized = question.strip().lower()
    normalized = normalized.rstrip("!.,? ")

    responses = {
        # Greetings
        "hi": "Hello! How can I help you today?",
        "hello": "Hello! How can I help you today?",
        "hey": "Hey! What can I help you with?",
        "good morning": "Good morning! How can I help you today?",
        "good afternoon": "Good afternoon! How can I help you today?",
        "good evening": "Good evening! How can I help you today?",
        "good night": "Good night! Take care.",

        # General conversation
        "how are you": "I'm doing well, thank you! What would you like to discuss?",
        "how are you doing": "I'm doing well, thank you! How are you?",
        "how is everything": "Everything is going well! What can I help you with?",
        "how's everything": "Everything is going well! What can I help you with?",
        "what are you doing": "I'm here and ready to help you with anything related to Agentic AI!",
        "what are you up to": "I'm here to help! What would you like to explore?",
        "what's up": "Hey! I'm ready to help. What's on your mind?",
        "whats up": "Hey! I'm ready to help. What's on your mind?",
        "what is up": "Hey! I'm ready to help. What's on your mind?",
        "how is it going": "It's going well! What can I help you with?",
        "nice to meet you": "Nice to meet you too! How can I help?",

        # Identity and capabilities
        "who are you": "I'm BotPlayer, an AI assistant specializing in Agentic AI and related topics.",
        "tell me about yourself": "I'm BotPlayer, an AI assistant specializing in Agentic AI. I can help explain AI concepts and answer questions using my knowledge base.",
        "can you help me": "Of course! Ask me a question, and I'll do my best to help.",
        "what can you do": "I can answer questions about Agentic AI and related topics using my knowledge base.",

        # Thanks
        "thank you": "You're welcome!",
        "thanks": "You're welcome!",
        "thanks a lot": "You're very welcome! Happy to help.",

        # Farewells
        "bye": "Goodbye! Feel free to come back whenever you need help.",
        "goodbye": "Goodbye! Feel free to come back whenever you need help.",
        "see you": "See you later! Feel free to come back whenever you need help.",
    }

    return responses.get(normalized)
