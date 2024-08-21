# import os
# import time
# import hashlib
# import sqlite3
# import streamlit as st
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import UnstructuredURLLoader
# from langchain_community.vectorstores import FAISS
# from urllib.parse import urlparse
# import logging
# from logging.handlers import RotatingFileHandler
# from datetime import datetime, timedelta
# import traceback
# from langchain.globals import set_verbose
# # from dotenv import load_dotenv
# # # Load environment variables
# # load_dotenv()

# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# # Set up logging
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# handler = RotatingFileHandler('finobot.log', maxBytes=10000, backupCount=3)
# logger.addHandler(handler)



# # Database setup
# conn = sqlite3.connect('finobot.db')
# c = conn.cursor()
# c.execute('''CREATE TABLE IF NOT EXISTS processed_urls
#              (url TEXT PRIMARY KEY, session_id TEXT)''')
# c.execute('''CREATE TABLE IF NOT EXISTS chat_history
#              (id INTEGER PRIMARY KEY, role TEXT, message TEXT, session_id TEXT)''')
# conn.commit()



# # Streamlit interface
# st.set_page_config(page_title="News Research Tool", page_icon="📰", layout="wide")
# st.markdown("""
#     <style>
#     .main-title {
#         font-size: 60px;
#         font-weight: bold;
#         color: #1E90FF;
#         text-align: center;
#         padding: 20px;
#         border-radius: 10px;
#         background-color: #F0F8FF;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#         margin-bottom:50px;
#     }
#     </style>
#     <h1 class="main-title">FinoBot 📈: Financial News & Analysis</h1>
#     """, unsafe_allow_html=True)




# # Sidebar Settings
# st.sidebar.markdown('<hr style="border:1px solid #ff4b4b; margin-top:2px; margin-bottom:2px;">', unsafe_allow_html=True)
# st.sidebar.markdown("""
#     <style>
#     .sidebar-title {
#         font-size: 40px;
#         font-weight: bold;
#         color: #1E90FF;
#         margin-bottom: 20px;
#     }
#     .sidebar-section {
#         background-color: #F0F8FF;
#         padding: 15px;
#         border-radius: 10px;
#         margin-bottom: 20px;
#         margin-top:5px;
#     }
#     </style>
#     <div class="sidebar-title">⚙️ Settings</div>
#     """, unsafe_allow_html=True)
# st.sidebar.markdown('<hr style="border:1px solid #ff4b4b;margin-top:2px;">', unsafe_allow_html=True)


# # Session management
# def get_session_id():
#     if 'session_id' not in st.session_state:
#         st.session_state.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()
#     return st.session_state.session_id

# session_id = get_session_id()


# # Initialize session state
# if 'vectorstore' not in st.session_state:
#     st.session_state.vectorstore = None
# if 'last_query_time' not in st.session_state:
#     st.session_state.last_query_time = datetime.min



# # Finobot Usage Guide
# st.markdown("""
#     <style>
#     .sidebar .sidebar-content {
#         background-color: #f0f8ff;
#     }
#     .sidebar .sidebar-content .block-container {
#         padding-top: 2rem;
#         padding-bottom: 2rem;
#     }
#     </style>
# """, unsafe_allow_html=True)

# st.sidebar.markdown(f""" <h1 style="font-size:30px; color:#FFD700;">Quick Guide 🚀 </h1>""", unsafe_allow_html=True)
# st.sidebar.markdown("""
# 1. 📥 **Input:** Enter up to 3 article URLs.
# 2. ⚙️ **Configure:** Set Chunk Size and Temperature.
# 3. ▶️ **Process:** Click "Process URLs".
# 4. ❓ **Query:** Ask your question in main area.
# 5. 📊 **Review:** Examine answer.
# 6. 🔄 **Manage:** Clear data or export history as needed.

# **Note: Process URLs before querying.**
# """)


# # Sample links
# st.sidebar.markdown('<hr style="border:1px solid #ff4b4b;">', unsafe_allow_html=True)
# st.sidebar.markdown("""
#         <h1 style="font-size:30px; color:#FFD700;">Sample Links 🌐 </h1>
#         <p style="font-size:20px; color:#4169e1;">Sample links for testing purpose 📄 </p>
#         <ul>
#             <li><a href="https://www.moneycontrol.com/news/podcast/can-markets-sustain-its-rally-after-mounting-to-new-highs-market-minutes-12781742.html" style="color: red;">🔗 Sample Link 1</a></li>
#             <li><a href="https://www.moneycontrol.com/news/business/markets/govt-may-raise-stcg-above-20-in-future-official-12782916.html" style="color: red;">🔗 Sample Link 2</a></li>
#             <li><a href="https://www.moneycontrol.com/news/business/itr-filing-last-date-income-tax-deadline-extension-july-2024-live-news-updates-liveblog-12782752.html" style="color: red;">🔗 Sample Link 3</a></li>
#         </ul>
 
#     """, unsafe_allow_html=True)



# # URL section
# st.sidebar.markdown('<hr style="border:1px solid #ff4b4b;">', unsafe_allow_html=True)
# st.sidebar.markdown(f"""<h1 style="font-size:30px; color:#FFD700;">URLs Section 🔗</h1>""", unsafe_allow_html=True)
# # st.sidebar.markdown('<h3 style="color: #1E90FF;">🔗 URLs Section</h3>', unsafe_allow_html=True)

# def validate_url(url):
#     try:
#         result = urlparse(url)
#         return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
#     except:
#         return False


# # URL inputs
# urls = []
# for i in range(3):
#     url = st.sidebar.text_input(f"URL {i+1}")
#     if url:
#         if validate_url(url):
#             urls.append(url)
#         else:
#              st.sidebar.warning(f"⚠️ URL {i+1} is not valid. Please enter a valid http or https URL.")
# st.sidebar.markdown('</div>', unsafe_allow_html=True)




# # Customization options
# st.sidebar.markdown('<hr style="border:1px solid #ff4b4b;">', unsafe_allow_html=True)
# st.sidebar.markdown(f"""<h1 style="font-size:30px; color:#FFD700;">Customization 🛠️</h1>""", unsafe_allow_html=True)

# chunk_size = st.sidebar.slider("📏 Chunk Size", min_value=100, max_value=2000, value=1000, step=100)
# temperature = st.sidebar.slider("🌡️ Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

# # process_url_clicked = st.sidebar.button("Process URLs")
# process_url_clicked = st.sidebar.button("▶️ Process URLs", key="process_urls")





# # Main content area
# main_placeholder = st.empty()


# # Initialize LLM
# try:
#     llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature, max_tokens=500)
# except Exception as e:
#     st.error(f"Error initializing LLM: {str(e)}")
#     logger.error(f"LLM initialization error: {str(e)}\n{traceback.format_exc()}")
#     st.stop()

# def process_urls(urls):
#     new_urls = [url for url in urls if url not in [row[0] for row in c.execute("SELECT url FROM processed_urls WHERE session_id=?", (session_id,))]]
    
#     if not new_urls:
#         st.warning("No new valid URLs to process.")
#         return

#     try:
#         with st.spinner('Processing URLs...'):
#             loader = UnstructuredURLLoader(urls=new_urls)
#             data = loader.load()
            
#             text_splitter = RecursiveCharacterTextSplitter(
#                 separators=['\n\n', '\n', '.', ','],
#                 chunk_size=chunk_size
#             )
#             docs = text_splitter.split_documents(data)
            
#             try:
#                 embeddings = OpenAIEmbeddings()
#             except Exception as e:
#                 st.warning("Failed to initialize OpenAI embeddings. Falling back to default embeddings.")
#                 logger.warning(f"OpenAI embeddings initialization failed: {str(e)}")
#                 embeddings = FAISS.default_embeddings()

#             vectorstore = FAISS.from_documents(docs, embeddings)
            
#             if st.session_state.vectorstore:
#                 st.session_state.vectorstore.merge_from(vectorstore)
#             else:
#                 st.session_state.vectorstore = vectorstore
            
#             for url in new_urls:
#                 c.execute("INSERT OR REPLACE INTO processed_urls (url, session_id) VALUES (?, ?)", (url, session_id))
#             conn.commit()
        
#         st.success(f"Processed {len(new_urls)} new URLs.")
#     except Exception as e:
#         st.error(f"An error occurred while processing URLs: {str(e)}")
#         logger.error(f"URL processing error: {str(e)}\n{traceback.format_exc()}")

# if process_url_clicked:
#     process_urls(urls)


# st.markdown("""
#     <style>
#     .stTextInput > div > div > input {
#         font-size: 1.1rem;
#         padding: 10px;
#         border-radius: 5px;
#         border: 2px solid #1E90FF;
#     }
#     .stButton > button {
#         background-color: #1E90FF;
#         color: white;
#         font-size: 1.1rem;
#         padding: 10px 20px;
#         border-radius: 5px;
#         border: none;
#     }
#     .stButton > button:hover {
#         background-color: #4169E1;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# st.markdown(f"""<h1 style="font-size:40px; color:#FFD700; margin-bottom: 1px;">❓ Ask a question about the processed news articles:</h1>""", unsafe_allow_html=True)
# st.markdown('<hr style="border:1px solid #ff4b4b;margin-top:2px;">', unsafe_allow_html=True)
# query = st.text_input('-----------------------------------------------------')
# ask_button = st.button('🔍 Ask Question')


# if ask_button and query:
#     if datetime.now() - st.session_state.last_query_time < timedelta(seconds=5):
#         st.warning("Please wait a few seconds before submitting another query.")
#     elif st.session_state.vectorstore is not None:
#         try:
#             qa_with_sources_chain = RetrievalQAWithSourcesChain.from_chain_type(
#                 llm=llm,
#                 chain_type="stuff",
#                 retriever=st.session_state.vectorstore.as_retriever()
#             )
            
#             with st.spinner('Searching for an answer...'):
#                 answer = qa_with_sources_chain.invoke({"question": query})
            
#             st.header("💡 Answer")
#             st.write(answer['answer'])
            
#             # st.header("📚 Sources")
#             # sources = answer['sources'].split('\n')
#             # for source in sources:
#             #     if source.strip():
#             #         st.markdown(f"- {source}")


            
#             # Add to chat history
#             c.execute("INSERT INTO chat_history (role, message, session_id) VALUES (?, ?, ?)", ("You", query, session_id))
#             c.execute("INSERT INTO chat_history (role, message, session_id) VALUES (?, ?, ?)", ("AI", answer['answer'], session_id))
#             conn.commit()

#             st.session_state.last_query_time = datetime.now()
#         except Exception as e:
#             st.error(f"An error occurred while processing your question: {str(e)}")
#             logger.error(f"Question processing error: {str(e)}\n{traceback.format_exc()}")
#     else:
#         st.error("Please process URLs first before asking questions.")
# elif ask_button:
#     st.warning("Please enter a question.")


# # Display Chat History
# st.markdown("""
#     <style>
#     .chat-history {
#        /* # background-color: #F0F8FF;*/
#         padding: 20px;
#         border-radius: 10px;
#         margin-top: 20px;
#     }
#     .chat-history h2 {
#         color: #FFD700;
#         margin-bottom: 1px;
#         font-size:40px;
#     }
#     </style>
#     <div class="chat-history">
#     <h2>Chat History 📜</h2>
    
#     </div>
#     <hr style="border:1px solid #ff4b4b;margin-top:1px;">
#     """, unsafe_allow_html=True)

# for row in c.execute("SELECT role, message FROM chat_history WHERE session_id=? ORDER BY id", (session_id,)):
#     st.markdown(f"**{row[0]}:** {row[1]}")


# # Clear data button
# if st.sidebar.button("🗑️ Clear Processed Data", key="clear_data"):
# # if st.sidebar.button("Clear Processed Data"):
#     try:
#         st.session_state.vectorstore = None
#         c.execute("DELETE FROM processed_urls WHERE session_id=?", (session_id,))
#         c.execute("DELETE FROM chat_history WHERE session_id=?", (session_id,))
#         conn.commit()
#         st.sidebar.success("All processed data has been cleared. You can now start fresh.")
#     except Exception as e:
#         st.error(f"Error clearing data: {str(e)}")
#         logger.error(f"Data clearing error: {str(e)}\n{traceback.format_exc()}")




# # Display processed URLs
# processed_urls = [row[0] for row in c.execute("SELECT url FROM processed_urls WHERE session_id=?", (session_id,))]
# if processed_urls:
#     st.sidebar.markdown(f"""<h1 style="font-size:20px; color:#ff4b4b;">Processed URLs</h1>""", unsafe_allow_html=True)
#     # st.sidebar.header("Processed URLs")
#     for url in processed_urls:
#         st.sidebar.markdown(f"- {url}")



# # Export chat history
# # Styling for the chat history section
# st.markdown("""
#     <style>
#     .chat-export-section {
#         background-color: #f0f8ff;
#         padding: 20px;
#         border-radius: 10px;
#         margin-top: 30px;
#         color:#1e90ff;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#     }
#     .chat-export-title {
#         color: #ff4b4b;
#         font-size: 24px;
#         margin-bottom: 15px;
#     }
#     .stDownloadButton>button {
#         background-color: #1e90ff;
#         color: white;
#         padding: 10px 20px;
#         border-radius: 5px;
#         border: none;
#         font-size: 16px;
#         transition: background-color 0.3s;
#     }
#     .stDownloadButton>button:hover {
#         background-color: #4169e1;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Chat history export section
# st.markdown('<h3 class="chat-export-title">📥 Export Chat History</h3>', unsafe_allow_html=True)

# if st.button("Prepare Export 🔄"):
#     try:
#         with st.spinner("Preparing chat history..."):
#             chat_history = c.execute("SELECT role, message FROM chat_history WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
#             chat_history_text = "\n\n".join([f"{'🧑‍💼' if role == 'You' else '🤖'} {role}:\n{message}" for role, message in chat_history])
        
#         st.success("Chat history prepared successfully! ✅")
#         st.download_button(
#             label="📥 Download Chat History",
#             data=chat_history_text,
#             file_name="finobot_chat_history.txt",
#             mime="text/plain",
#             key="download_chat"
#         )
#     except Exception as e:
#         st.error(f"🚫 Error exporting chat history: {str(e)}")
#         logger.error(f"Chat history export error: {str(e)}\n{traceback.format_exc()}")

# st.markdown('', unsafe_allow_html=True)



# # Feedback Section
# st.sidebar.markdown('<hr style="border:1px solid #ff4b4b;">', unsafe_allow_html=True)
# st.sidebar.markdown(f"""
#     <h1 style="font-size:30px; color:#FFD700;">Feedback 📝</h1>
#     <p>We value your feedback! 😊</p>
# """, unsafe_allow_html=True)

# feedback = st.sidebar.text_area("Please leave your feedback here:")
# if st.sidebar.button("Submit Feedback"):
#     st.sidebar.success("Thank you for your feedback! 👍")



# # Close database connection
# conn.close()


# # Footer
# st.markdown("---")
# st.markdown('<p style="text-align: center;">Created with ❤️ using Streamlit and LangChain</p>', unsafe_allow_html=True)
































































import os
import time
import hashlib
import sqlite3
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from urllib.parse import urlparse
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
import traceback
import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import base64

# Download NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('finobot.log', maxBytes=10000, backupCount=3)
logger.addHandler(handler)

# Environment setup
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Database setup
conn = sqlite3.connect('finobot.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS processed_urls
             (url TEXT PRIMARY KEY, session_id TEXT, content TEXT, sentiment REAL)''')
c.execute('''CREATE TABLE IF NOT EXISTS chat_history
             (id INTEGER PRIMARY KEY, role TEXT, message TEXT, session_id TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
c.execute('''CREATE TABLE IF NOT EXISTS user_feedback
             (id INTEGER PRIMARY KEY, feedback TEXT, session_id TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
# Add sentiment column if it doesn't exist
c.execute("PRAGMA table_info(processed_urls)")
columns = [info[1] for info in c.fetchall()]
if 'content' not in columns:
    c.execute("ALTER TABLE processed_urls ADD COLUMN content TEXT")
if 'sentiment' not in columns:
    c.execute("ALTER TABLE processed_urls ADD COLUMN sentiment REAL")
conn.commit()

# Streamlit configuration
st.set_page_config(page_title="FinoBot: AI-Powered Financial Analyst", page_icon="📊", layout="wide")





st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;700&family=Inter:wght@300;400;600&display=swap');
    
    :root {
        --primary-color: #00F5FF;
        --secondary-color: #FF00E4;
        --accent-color: #FFD700;
        --bg-color: #0A0E17;
        --text-color: #E0E0E0;
        --card-bg: #141C2F;
    }
    
    body {
        color: var(--text-color);
        background-color: var(--bg-color);
        font-family: 'Inter', sans-serif;
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(0, 245, 255, 0.1) 0%, transparent 20%),
            radial-gradient(circle at 90% 80%, rgba(255, 0, 228, 0.1) 0%, transparent 20%);
        background-attachment: fixed;
    }
    
    .stApp {
        background: transparent;
    }
    
    h1, h2, h3 {
        font-family: 'Exo 2', sans-serif;
        color: var(--primary-color);
        text-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    .stButton > button {
        font-family: 'Exo 2', sans-serif;
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: var(--bg-color);
        font-weight: 700;
        border-radius: 30px;
        border: none;
        padding: 15px 30px;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0, 245, 255, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 20px rgba(255, 0, 228, 0.6);
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transform: rotate(45deg);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { left: -50%; top: -50%; }
        100% { left: 150%; top: 150%; }
    }
    
    .stTextInput > div > div > input, 
    .stSelectbox > div > div > select, 
    .stTextArea > div > div > textarea {
        font-family: 'Inter', sans-serif;
        background-color: var(--card-bg);
        color: var(--text-color);
        border-radius: 15px;
        border: 2px solid var(--primary-color);
        padding: 12px;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus, 
    .stSelectbox > div > div > select:focus, 
    .stTextArea > div > div > textarea:focus {
        border-color: var(--secondary-color);
        box-shadow: 0 0 15px rgba(255, 0, 228, 0.5);
    }
    
    .stTab {
        font-family: 'Exo 2', sans-serif;
        background-color: var(--card-bg);
        color: var(--text-color);
        font-weight: 600;
        border-radius: 10px 10px 0 0;
        border: 2px solid var(--primary-color);
        border-bottom: none;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stTab[aria-selected="true"] {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: var(--bg-color);
    }
    
    .stDataFrame {
        font-family: 'Inter', sans-serif;
        border: 2px solid var(--primary-color);
        border-radius: 15px;
        overflow: hidden;
    }
    
    .stDataFrame thead {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: var(--bg-color);
        font-family: 'Exo 2', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stDataFrame tbody tr:nth-of-type(even) {
        background-color: rgba(20, 28, 47, 0.7);
    }
    
    .stAlert {
        font-family: 'Inter', sans-serif;
        background-color: var(--card-bg);
        color: var(--text-color);
        border-radius: 15px;
        border: 2px solid var(--primary-color);
    }
    
    .stProgress > div > div > div > div {
        background-color: var(--primary-color);
    }
    
    .stSlider > div > div > div > div {
        color: var(--primary-color);
        font-family: 'Exo 2', sans-serif;
    }
    
    .css-1cpxqw2 {
        background-color: var(--card-bg);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0, 245, 255, 0.2);
        transition: all 0.3s ease;
        border: 2px solid transparent;
        background-clip: padding-box;
    }
    
    .css-1cpxqw2:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(255, 0, 228, 0.3);
        border-color: var(--secondary-color);
    }
    
    @keyframes glow {
        0% { box-shadow: 0 0 5px var(--primary-color); }
        50% { box-shadow: 0 0 20px var(--primary-color), 0 0 30px var(--secondary-color); }
        100% { box-shadow: 0 0 5px var(--primary-color); }
    }
    
    .glow-effect {
        animation: glow 2s infinite;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-color);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(var(--primary-color), var(--secondary-color));
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(var(--secondary-color), var(--primary-color));
    }
    
    /* Chat bubbles */
    .chat-bubble {
        padding: 10px 15px;
        border-radius: 20px;
        margin-bottom: 10px;
        max-width: 80%;
        clear: both;
    }
    .user-bubble {
        background-color: rgba(0, 245, 255, 0.2);
        float: right;
    }
    .ai-bubble {
        background-color: rgba(255, 0, 228, 0.2);
        float: left;
    }
</style>
""", unsafe_allow_html=True)


## Header with logo
st.markdown(
    '''
    <h1 style="text-align: center;">
        <img src="data:image/png;base64,{}" width="150" />
        FinoBot: AI-Powered Financial Analyst
    </h1>
    '''.format(base64.b64encode(open("New_logo.png", "rb").read()).decode('utf-8')),
    unsafe_allow_html=True
)


# # Animated title
# st.markdown("""
# <h1 class="neon-glow">FinoBot 📊: AI-Powered Financial Analyst</h1>
# """, unsafe_allow_html=True)

# Session management
def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()
    return st.session_state.session_id

session_id = get_session_id()

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'last_query_time' not in st.session_state:
    st.session_state.last_query_time = datetime.min
if 'processed_urls' not in st.session_state:
    st.session_state.processed_urls = []

# Helper functions
def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
    except:
        return False

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

def process_urls(urls):
    new_urls = [url for url in urls if url not in st.session_state.processed_urls]
    
    if not new_urls:
        st.warning("No new valid URLs to process.")
        return

    try:
        with st.spinner('Processing URLs...'):
            loader = UnstructuredURLLoader(urls=new_urls)
            data = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=st.session_state.chunk_size
            )
            docs = text_splitter.split_documents(data)
            
            try:
                embeddings = OpenAIEmbeddings()
            except Exception as e:
                st.warning("Failed to initialize OpenAI embeddings. Falling back to default embeddings.")
                logger.warning(f"OpenAI embeddings initialization failed: {str(e)}")
                embeddings = FAISS.default_embeddings()

            vectorstore = FAISS.from_documents(docs, embeddings)
            
            if st.session_state.vectorstore:
                st.session_state.vectorstore.merge_from(vectorstore)
            else:
                st.session_state.vectorstore = vectorstore
            
            for url, doc in zip(new_urls, data):
                content = doc.page_content
                sentiment = get_sentiment(content)
                c.execute("INSERT OR REPLACE INTO processed_urls (url, session_id, content, sentiment) VALUES (?, ?, ?, ?)", 
                          (url, session_id, content, sentiment))
                st.session_state.processed_urls.append(url)
            conn.commit()
        
        st.success(f"Processed {len(new_urls)} new URLs.")
    except Exception as e:
        st.error(f"An error occurred while processing URLs: {str(e)}")
        logger.error(f"URL processing error: {str(e)}\n{traceback.format_exc()}")

def ask_question(query):
    if datetime.now() - st.session_state.last_query_time < timedelta(seconds=5):
        st.warning("Please wait a few seconds before submitting another query.")
        return

    if st.session_state.vectorstore is not None:
        try:
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=st.session_state.temperature, max_tokens=500)
            qa_with_sources_chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.vectorstore.as_retriever()
            )
            
            with st.spinner('Analyzing financial data...'):
                answer = qa_with_sources_chain.invoke({"question": query})
            
            st.markdown("### 💡 FinoBot's Analysis:")
            st.markdown(f'<div class="chat-bubble ai-bubble">{answer["answer"]}</div>', unsafe_allow_html=True)
            
            st.markdown("### 📚 Sources:")
            sources = answer['sources'].split('\n')
            for source in sources:
                if source.strip():
                    st.markdown(f"- {source}")
            
            # Add to chat history
            c.execute("INSERT INTO chat_history (role, message, session_id) VALUES (?, ?, ?)", ("AI", answer['answer'], session_id))
            conn.commit()

            st.session_state.last_query_time = datetime.now()
        except Exception as e:
            st.error(f"An error occurred while processing your question: {str(e)}")
            logger.error(f"Question processing error: {str(e)}\n{traceback.format_exc()}")
        else:
            st.error("Please process URLs first before asking questions.")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Home", "📥 Data Input", "⚙️ Settings", "💬 Chat History", "📊 Analytics"])

with tab1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header("FinoBot Dashboard")
    st.write("""
    Welcome to FinoBot, your AI-powered financial news analyzer and advisor. Here's how to get started:
    1. 📥Input financial news article URLs in the 'Data Input' tab.
    2. ⚙️Customize your experience in the 'Settings' tab.
    3. ▶️Process the URLs and ask questions about the financial content.
    4. 🔄Review your conversation history in the 'Chat History' tab.
    5. 📊Explore in-depth data visualizations in the 'Analytics' tab.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="custom-card neon-glow">', unsafe_allow_html=True)
        st.metric("Processed URLs", len(st.session_state.processed_urls))
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="custom-card neon-glow">', unsafe_allow_html=True)
        # Check if 'sentiment' column exists
        c.execute("PRAGMA table_info(processed_urls)")
        columns = [info[1] for info in c.fetchall()]
        if 'sentiment' in columns:
            avg_sentiment = c.execute("SELECT AVG(sentiment) FROM processed_urls WHERE session_id=?", (session_id,)).fetchone()[0]
            st.metric("Average Sentiment", f"{avg_sentiment:.2f}" if avg_sentiment else "N/A")
        else:
            st.metric("Average Sentiment", "N/A")
            st.warning("Sentiment analysis not available. Please reprocess your URLs.")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="custom-card neon-glow">', unsafe_allow_html=True)
        chat_count = c.execute("SELECT COUNT(*) FROM chat_history WHERE session_id=?", (session_id,)).fetchone()[0]
        st.metric("Chat Interactions", chat_count)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Start Analyzing", key="get_started"):
        st.session_state.active_tab = "Data Input"

    # Add a section to show recent processed URLs
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("Recently Processed URLs")
    recent_urls = c.execute("SELECT url FROM processed_urls WHERE session_id=? ORDER BY rowid DESC LIMIT 5", (session_id,)).fetchall()
    if recent_urls:
        for url in recent_urls:
            st.write(f"- {url[0]}")
    else:
        st.write("No URLs processed yet. Head to the 'Data Input' tab to get started!")
    st.markdown('</div>', unsafe_allow_html=True)

    # Add a quick sentiment overview if sentiment data is available
    if 'sentiment' in columns:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("Quick Sentiment Overview")
        sentiment_counts = c.execute("""
            SELECT 
                CASE 
                    WHEN sentiment > 0 THEN 'Positive'
                    WHEN sentiment < 0 THEN 'Negative'
                    ELSE 'Neutral'
                END as sentiment_category,
                COUNT(*) as count
            FROM processed_urls
            WHERE session_id=?
            GROUP BY sentiment_category
        """, (session_id,)).fetchall()
        
        if sentiment_counts:
            fig = px.pie(
                values=[count for _, count in sentiment_counts],
                names=[category for category, _ in sentiment_counts],
                title="Sentiment Distribution of Processed Articles"
            )
            fig.update_layout(
                font=dict(color="#00F5FF"),
                paper_bgcolor="#141C2F",
                plot_bgcolor="#0A0E17"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No sentiment data available yet. Process some URLs to see the sentiment distribution.")
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header("Data Input")
    urls = []
    for i in range(3):
        url = st.text_input(f"Financial News URL {i+1}", key=f"url_input_{i}")
        if url and validate_url(url):
            urls.append(url)
        elif url:
            st.warning(f"⚠️ URL {i+1} is not valid. Please enter a valid http or https URL.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Process URLs", key="process_urls"):
            process_urls(urls)
    
    with col2:
        if st.button("🗑️ Clear Processed Data", key="clear_data"):
            st.session_state.vectorstore = None
            st.session_state.processed_urls = []
            c.execute("DELETE FROM processed_urls WHERE session_id=?", (session_id,))
            c.execute("DELETE FROM chat_history WHERE session_id=?", (session_id,))
            conn.commit()
            st.success("All processed data has been cleared. You can now start fresh.")

    st.subheader("Sample Financial News Links")
    st.markdown("""
    - [CNBC: Market Analysis](https://www.cnbc.com/markets/)
    - [Bloomberg: Financial News](https://www.bloomberg.com/markets)
    - [Reuters: Business News](https://www.reuters.com/business/)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header("FinoBot Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.chunk_size = st.slider("📏 Text Chunk Size", min_value=100, max_value=2000, value=1000, step=100)
    with col2:
        st.session_state.temperature = st.slider("🌡️ AI Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    st.info("Chunk Size: Larger values process more context at once. Temperature: Higher values make AI responses more creative, lower values make them more focused.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header("Chat History")
    chat_history = c.execute("SELECT role, message FROM chat_history WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
    for role, message in chat_history:
        if role == 'You':
            st.markdown(f'<div class="chat-bubble user-bubble">🧑‍💼 You: {message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble ai-bubble">🤖 FinoBot: {message}</div>', unsafe_allow_html=True)
    
    if st.button("📥 Export Chat History"):
        chat_history_text = "\n\n".join([f"{'🧑‍💼 You:' if role == 'You' else '🤖 FinoBot:'}\n{message}" for role, message in chat_history])
        st.download_button(
            label="Download Chat History",
            data=chat_history_text,
            file_name="finobot_chat_history.txt",
            mime="text/plain"
        )
    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header("Financial Analytics")
    
    # Sentiment Analysis
    sentiments = c.execute("SELECT sentiment FROM processed_urls WHERE session_id=?", (session_id,)).fetchall()
    if sentiments:
        fig = px.histogram(x=[s[0] for s in sentiments], nbins=20, 
                           labels={'x': 'Sentiment', 'y': 'Count'},
                           title='Sentiment Distribution of Processed Articles')
        fig.update_layout(
            font=dict(color="#00F5FF"),
            paper_bgcolor="#141C2F",
            plot_bgcolor="#0A0E17"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Word Cloud
    if st.session_state.processed_urls:
        st.subheader("Word Cloud of Processed Articles")
        all_text = ' '.join(c.execute("SELECT content FROM processed_urls WHERE session_id=?", (session_id,)).fetchall()[0])
        wordcloud = WordCloud(width=800, height=400, background_color='#0A0E17', colormap='viridis').generate(all_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_facecolor('#0A0E17')
        fig.patch.set_facecolor('#0A0E17')
        st.pyplot(fig)
    
    # Time Series Analysis (placeholder)
    st.subheader("Financial Time Series Analysis")
    date_range = pd.date_range(end=pd.Timestamp.now(), periods=30)
    fake_data = pd.DataFrame({
        'Date': date_range,
        'Stock Price': np.random.randn(30).cumsum() + 100
    })
    fig = px.line(fake_data, x='Date', y='Stock Price', title='Sample Stock Price Trend')
    fig.update_layout(
        font=dict(color="#00F5FF"),
        paper_bgcolor="#141C2F",
        plot_bgcolor="#0A0E17"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header("Quick Actions")
    query = st.text_input("Ask FinoBot a question:")
    if st.button("🔍 Analyze"):
        ask_question(query)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.sidebar.header("We value your feedback! 🌟")

    feedback = st.sidebar.text_area("📝 Leave Feedback", "Share your thoughts and suggestions here...")
    rating = st.sidebar.slider("📊 Rate Your Experience", 1, 5, 3)
    suggestions = st.sidebar.text_input("🗣️ Share Suggestions", "Any ideas for new features or improvements?")
    issue = st.sidebar.text_area("🧩 Report Issues", "Describe any problems you encountered...")

    if st.sidebar.button("Submit Feedback"):
        st.sidebar.success("Thank you for your feedback! 👍")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: #00F5FF;">FinoBot: Empowering financial decisions with AI | Created with ❤️ using Streamlit and LangChain</p>', unsafe_allow_html=True)

# Close database connection
conn.close()




