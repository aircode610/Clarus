"""
Run the Clarus Streamlit UI

This script starts the Streamlit web interface for the Clarus application.
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application."""
    print("🚀 Starting Clarus Streamlit UI...")
    print("📝 Clarus - Intelligent Document Structuring System")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit installed successfully")
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY environment variable not set.")
        print("   Set your OpenAI API key to use the AI features.")
        print("   You can still explore the UI, but AI features won't work.")
        print()
    
    print("🌐 Starting web interface...")
    print("   The app will open in your default browser.")
    print("   If it doesn't open automatically, go to: http://localhost:8501")
    print()
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Clarus UI stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
