import streamlit as st
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.readers.web import SimpleWebPageReader
import openai
import requests
import tempfile


user_icon = "./icons/student.png"
assistant_icon = "./icons/assistant1.png"
# Streamlit app configuration
st.set_page_config(
    page_title="Learner",
    page_icon="🤩",
    layout="centered",
    initial_sidebar_state="auto",
)


st.markdown(
    """
<style>
    .title-container {
    font-family: Arial, sans-serif;
    
    }
    .block-container {
        padding-top: 32px;
        padding-bottom: 32px;
        padding-left: 0;
        padding-right: 0;
    }

    .main-header {
        font-size: 24px;
    }

    /* Add your chat message styles here */
    .chat-message {
        border-radius: 25px;
        padding: 10px;
        margin-bottom: 10px; /* Add space below each message */
        color: white;
    }

    .user-message {
        background-color: #002651; /* Light blue background for user messages */
        color: white;
    }

    .assistant-message {
        background-color: #581b98; /* Light purple background for assistant messages */
    }

    /* Additional styles to space out audio elements */
    .audio-container {
        margin-top: 5px; /* Add space above the audio player */
        margin-bottom: 15px; /* Add space below the audio player */
    }
</style>
""",
    unsafe_allow_html=True,
)


# Set OpenAI API key
openai.api_key = st.secrets.openaikey

# App title and info
st.title("Learning Helper")

with st.container() as title_container:
    st.info(
        "Ability to query all your documents!",
    )

st.image("./icons/logoass.png")


# Function to load and index the knowledge base
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(
        "Loading and indexing the knowledge base – hang tight! This should take 1-2 minutes."
    ):
        # Load documents from a directory
        directory_reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs_from_dir = directory_reader.load_data()

        # Load documents from web pages
        web_page_urls = [
            "https://www.uc.edu/content/dam/uc/ce/docs/OLLI/Page%20Content/ARTIFICIAL%20INTELLIGENCEr.pdf",
        ]
        web_reader = SimpleWebPageReader(html_to_text=True)
        docs_from_web = web_reader.load_data(web_page_urls)

        # Combine documents from both sources
        docs = docs_from_dir + docs_from_web

        # Create a service context with OpenAI model
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(
                model="gpt-3.5-turbo",
                temperature=0.5,
                system_prompt="you are a learning assistant- provide the content in an ordered manner",
            )
        )
        # Index documents
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index


index = load_data()

# Initialize chat messages history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about your documents!",
            "audio": None,
        }
    ]

# Initialize chat engine in session state
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(verbose=True)


# Function to call TTS API
def text_to_speech(text):
    api_url = "https://api.elevenlabs.io/v1/text-to-speech/tzYzGF8QPOE4A9WGNE5h"
    headers = {
        "xi-api-key": st.secrets.elevenlabsapikey,
        "Content-Type": "application/json",
    }

    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.75, "similarity_boost": 0.75},
    }
    response = requests.post(api_url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.content
    else:
        st.error("Failed to get audio from ElevenLabs: " + response.text)
        return None


def display_chat_messages(messages):
    for message in messages:
        # Determine the CSS class based on the message role
        message_class = (
            "user-message" if message["role"] == "user" else "assistant-message"
        )

        # Display the message with styling
        col1, col2 = st.columns([1, 18], gap="small")
        with col1:
            st.image(message.get("icon", assistant_icon), width=30)  # Display the icon

        with col2:
            # Wrap the message content in a div with the appropriate class for styling
            st.markdown(
                f"""
            <div class="chat-message {message_class}">
                {message["content"]}
            </div>
            """,
                unsafe_allow_html=True,
            )

            # If there's audio, wrap it in a div with the audio-container class for spacing
            if message.get("audio"):
                st.markdown(
                    f"""
                <div class="audio-container">
                """,
                    unsafe_allow_html=True,
                )
                st.audio(message["audio"], format="audio/mp3")
                st.markdown(
                    f"""
                </div>
                """,
                    unsafe_allow_html=True,
                )


# Incorporate spinner around the response generation and TTS processing
if prompt := st.chat_input("Your question"):
    # Append the user message to the chat history first
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "icon": user_icon}
    )

    # Use spinner while waiting for the assistant's response and processing TTS
    with st.spinner("Thinking..."):
        response = st.session_state.chat_engine.chat(
            prompt
        )  # Generate the assistant's response
        audio_content = text_to_speech(
            response.response
        )  # Generate audio from the response
        audio_path = None
        if audio_content:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                fp.write(audio_content)
                fp.flush()
                audio_path = fp.name

    # Append the assistant's message and audio to the chat history after processing
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.response,
            "icon": assistant_icon,
            "audio": audio_path,
        }
    )

# Ensure to initialize `st.session_state.messages` at the beginning if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
display_chat_messages(st.session_state.messages)
