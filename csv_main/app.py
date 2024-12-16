from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_csv_agent
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configure Google Generative AI
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set!")

genai.configure(api_key=API_KEY)

# Define the model
model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    client=genai,
    temperature=0.1,
    top_k=10,
)

app = Flask(__name__)

# Define the folder for storing uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions for CSV files
ALLOWED_EXTENSIONS = {'csv'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create the 'uploads' directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        custom_prompt = PromptTemplate(
            input_variables=["input", "csv_filename"],
            template=(
                "You are an assistant skilled at analyzing CSV data and explaining findings "
                "in a human-like and easy-to-understand way. Please answer the following query "
                "based on the contents of {csv_filename}:\n{input}"
            )
        )
        title = request.form.get('title')
        desc = request.form.get('desc')
        csv_file = request.files.get('csvFile')

        # Handle the CSV file if uploaded
        if csv_file and allowed_file(csv_file.filename):
            # Save the file to the server
            csv_filename = os.path.join(app.config['UPLOAD_FOLDER'], csv_file.filename)
            csv_file.save(csv_filename)
            
            # Process the CSV file with Langchain
            try:
                # Create the agent with the uploaded CSV
                agent = create_csv_agent(
                    llm=model,
                    path=csv_filename,
                    verbose=True,
                    allow_dangerous_code=True,
                    prompt=custom_prompt
                )
                
                # Get the user's question
                user_question = request.form.get('user_question')
                
                if user_question.strip():  # Avoid empty or whitespace-only input
                    response = agent.run(user_question)
                    print(f'hi{print(response)}')
                    result = response  # Store the result to display it
                    
                else:
                    result = "Please enter a question about the CSV."
                
            except Exception as e:
                result = f"Error processing your question: {e}"
        
        return render_template('index.html', result=result)

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
