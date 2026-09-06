import speech_recognition as sr
import pyttsx3
import threading
import queue
from datetime import datetime
import json
import os
from typing import Optional, Dict, Any
import anthropic

# Initialize recognizer and text-to-speech engine
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

# Configure TTS
tts_engine.setProperty('rate', 150)  # Speed of speech
tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

class VoiceAIAgent:
    """Multi-tasking AI agent with real-time voice conversation support"""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize the Voice AI Agent
        
        Args:
            model: Claude model to use for processing
        """
        self.client = anthropic.Anthropic()
        self.model = model
        self.conversation_history = []
        self.task_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.is_running = False
        self.current_task = None
        
    def listen_to_audio(self, timeout: int = 10) -> Optional[str]:
        """
        Convert speech to text using Google Speech Recognition
        
        Args:
            timeout: Maximum time to listen for audio (in seconds)
            
        Returns:
            Transcribed text or None if speech recognition fails
        """
        try:
            print("\n🎤 Listening...")
            with sr.Microphone() as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=30)
            
            print("⏳ Processing speech...")
            text = recognizer.recognize_google(audio)
            print(f"✅ You said: {text}")
            return text
            
        except sr.RequestError as e:
            print(f"❌ API error: {e}")
            return None
        except sr.UnknownValueError:
            print("❌ Sorry, I couldn't understand what you said. Please try again.")
            return None
        except Exception as e:
            print(f"�� Error: {e}")
            return None
    
    def process_input(self, user_input: str) -> str:
        """
        Process user input using Claude AI
        
        Args:
            user_input: User's spoken input
            
        Returns:
            AI-generated response
        """
        try:
            # Add user message to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_input
            })
            
            print("\n🤖 Processing with AI...")
            
            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system="""You are a helpful multi-tasking AI assistant that can:
1. Answer questions and provide information
2. Help with task management and planning
3. Engage in creative conversations
4. Provide technical assistance
5. Assist with writing and editing

Be concise, helpful, and friendly. Keep responses relatively short for voice interaction.""",
                messages=self.conversation_history
            )
            
            # Extract response text
            assistant_message = response.content[0].text
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
            
        except Exception as e:
            print(f"❌ Error processing input: {e}")
            return "Sorry, I encountered an error processing your request."
    
    def speak_response(self, response_text: str) -> None:
        """
        Convert text to speech and play it
        
        Args:
            response_text: Text to convert to speech
        """
        try:
            print(f"\n🔊 AI Response: {response_text}")
            tts_engine.say(response_text)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"❌ Error in text-to-speech: {e}")
    
    def handle_conversation_turn(self) -> bool:
        """
        Handle one complete conversation turn:
        Listen -> Process -> Respond
        
        Returns:
            True if conversation should continue, False otherwise
        """
        try:
            # Step 1: Listen for user input
            user_input = self.listen_to_audio()
            
            if not user_input:
                return True  # Continue listening
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye', 'stop']:
                self.speak_response("Goodbye! Thanks for chatting with me.")
                return False
            
            # Step 2: Process input with AI
            response = self.process_input(user_input)
            
            # Step 3: Speak the response
            self.speak_response(response)
            
            return True  # Continue conversation
            
        except Exception as e:
            print(f"❌ Error in conversation turn: {e}")
            return True
    
    def start_conversation(self) -> None:
        """Start the main conversation loop"""
        try:
            self.is_running = True
            print("\n" + "="*60)
            print("🎙️  VOICE AI AGENT STARTED")
            print("="*60)
            print("Commands: Say 'exit', 'quit', or 'goodbye' to stop")
            print("="*60)
            
            # Greet the user
            greeting = "Hello! I'm your AI assistant. How can I help you today?"
            self.speak_response(greeting)
            
            # Main conversation loop
            while self.is_running:
                if not self.handle_conversation_turn():
                    break
            
            print("\n" + "="*60)
            print("👋 Conversation ended")
            print("="*60)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            self.speak_response("Session ended.")
        except Exception as e:
            print(f"❌ Fatal error: {e}")
        finally:
            self.is_running = False
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the conversation
        
        Returns:
            Dictionary containing conversation metadata and history
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "turn_count": len(self.conversation_history),
            "history": self.conversation_history
        }
    
    def save_conversation(self, filename: str = "conversation_log.json") -> None:
        """
        Save conversation history to a JSON file
        
        Args:
            filename: Name of the file to save to
        """
        try:
            summary = self.get_conversation_summary()
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"✅ Conversation saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving conversation: {e}")
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history = []
        print("✅ Conversation history cleared")


def main():
    """Main entry point"""
    try:
        # Create agent instance
        agent = VoiceAIAgent()
        
        # Start the conversation
        agent.start_conversation()
        
        # Save conversation log
        agent.save_conversation()
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")


if __name__ == "__main__":
    main()
