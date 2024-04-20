import dspy
from dspy import dsp
import os
from dspy.retrieve.weaviate_rm import WeaviateRM
from weaviate.classes.init import AdditionalConfig, Timeout
import weaviate
import json
import streamlit as st
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl
from tools import check_json, company_url, resume_into_json
import nltk
from PyPDF2 import PdfReader
import cohere

co_api_key = os.getenv("CO_API_KEY")
nltk.download('punkt')

# Weaviate client configuration
url = "https://internship-finder-52en6hka.weaviate.network"
apikey = os.getenv("WCS_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Connect to Weaviate
weaviate_client = weaviate.connect_to_wcs(
    cluster_url=url,  
    auth_credentials=weaviate.auth.AuthApiKey(apikey),
        headers={
        "X-OpenAI-Api-Key": openai_api_key  
    },additional_config=AdditionalConfig(
        timeout=Timeout(init=2, query=45, insert=120)  # Values in seconds
    ) 
    
)

cohere = dsp.Cohere(model='command-r-plus',api_key=co_api_key)

retriever_model = WeaviateRM("Internship", weaviate_client=weaviate_client)

dspy.settings.configure(lm=cohere,rm=retriever_model)
# Weaviate client configuration
st.title("Internship Finder")
my_bar = st.progress(0)

class JobListing(BaseModel):
    city: str
    date_published: datetime  # Assuming the date can be parsed into a datetime object
    apply_link: HttpUrl  # Pydantic will validate this is a valid URL
    company: str
    location: Optional[str]  # Assuming 'location' could be a string or None
    country: str
    name: str

class Out_Internship(BaseModel):
    output: list[JobListing] = Field(description="list of internships")  

def search_datbase(query):
    url = "https://internship-finder-52en6hka.weaviate.network"
    apikey = os.getenv("WCS_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

# Connect to Weaviate
    weaviate_client = weaviate.connect_to_wcs(
    cluster_url=url,  
    auth_credentials=weaviate.auth.AuthApiKey(apikey),
        headers={
        "X-OpenAI-Api-Key": openai_api_key  
    }  
    
    )
    questions = weaviate_client.collections.get("Internship")

    response = questions.query.hybrid(
        query=query,
        limit=10
    )

    interns = []

    # adding internships to list
    for item in response.objects:
        interns.append(item.properties) 
    
    
    context = json.dumps(interns)
    weaviate_client.close()
    return json.loads(context)

def check_resume(resume):
    if (resume != None):
        pdf_reader = PdfReader(resume)
        text=""
        for page_num in range(len(pdf_reader.pages)):
                
                page = pdf_reader.pages[page_num]
                
                # Extract text from the page
                text += page.extract_text()
    
    
    tokens = nltk.word_tokenize(text)
    
    # Check if the total character count of all tokens exceeds the limit
    total_length = sum(len(token) for token in tokens)
    if total_length >= 16000:
        return False  # Return False if the total length of tokens exceeds the limit

    tokens_to_check = ["summary", "skills", "experience", "projects", "education"]
    
    # Convert tokens to lower case for case-insensitive comparison
    text_tokens_lower = [token.lower() for token in tokens]

    # Check if any of the specified tokens are in the tokenized text
    tokens_found = [token for token in tokens_to_check if token.lower() in text_tokens_lower]

    # Return False if none of the specified tokens were found, True otherwise
    return bool(tokens_found)



class Internship_finder(dspy.Module):
    cohere = dsp.Cohere(model='command-r-plus',api_key=co_api_key)

    dspy.settings.configure(lm=cohere)
    def __init__(self):
        super().__init__()
        self.generate_query = [dspy.ChainOfThought(generate_query) for _ in range(3)]
        self.generate_analysis = dspy.Predict(generate_analysis,max_tokens=4000) 

    def forward(self, resume):
        #resume to pass as context 
        
        passages = []

        for hop in range(3):
            query = self.generate_query[hop](context=str(resume)).query
            info=search_datbase(query)
            passages.append(info)

        context = deduplicate(passages)  
        my_bar.progress(60,text="Doing Analysis")
            
        analysis = self.generate_analysis(resume=str(resume), context=context).output
              
        return analysis
    


def deduplicate(context):
        """
        Removes duplicate elements from the context list while preserving the order.
        
        Parameters:
        context (list): List containing context elements.
        
        Returns:
        list: List with duplicates removed.
        """
        json_strings = [json.dumps(d, sort_keys=True) for d in context]
    
        # Use a set to remove duplicate JSON strings
        unique_json_strings = set(json_strings)
    
        # Convert JSON strings back to dictionaries
        unique_dicts = [json.loads(s) for s in unique_json_strings]
        return unique_dicts

def check_answer(assessment_answer):
    if assessment_answer == "no":
        return False
    return True

def get_resume():
    with open('resume.json', 'r') as file: 
        resume = json.load(file)
     
    return resume



class generate_analysis(dspy.Signature):
    """
    Your Role:
    As a Matchmaking Manager, your expertise lies in connecting students with their dream internship opportunities. You are equipped with a student's resume and a list of potential internships. Your task is to meticulously analyze and identify the best matches by following specific criteria.


    Matching Criteria:


    Educational Background:


    Degree and Major: Aim for exact matches between the student's degree level and major and the requirements specified in the internships. Consider close alignments if exact matches are not available.
    Related Fields of Study: Treat closely related fields of study as potential matches. For instance, consider a student with a Computer Science major for IT or Software Engineering internship opportunities.
    Relevant Coursework: Give preference to internships that specifically seek coursework mentioned in the student's resume. For example, if an internship prefers candidates with advanced Data Structures knowledge, and the student has taken a relevant course, it strengthens the match.

    Skill and Experience Match:


    Required Skills: Look for strong overlaps between the technical skills on the student's resume and the required skills in the internship descriptions.
    Tools and Frameworks: Prioritize internships that specifically mention tools, programming languages, or frameworks that the student has hands-on experience with. Proficiency or project experience with required tools is a strong indicator of a good fit.
    Applied Skills: Value practical, hands-on project or work experience that demonstrates the application of required skills. For instance, if an internship seeks web development skills, and the student has a portfolio of built and deployed websites, it is a clear advantage.

    Project Relevance:


    Project Experience: Scrutinize the student's project portfolio to identify technical skills and expertise that align with the internships' requirements.
    AI/ML and Data Focus: Specifically, look for internships that seek experience or interest in AI/ML model development, data analysis, or similar fields. Keyword matches such as "machine learning," "data engineering," or "data-driven solutions" indicate potential alignments.
    Exclusion Criteria: Ensure that the internships do not include any mention of "research" in their titles, required skills, or descriptions.
    Practical Implementation: Favor internships that emphasize practical, hands-on experience in development, engineering, or application development roles over theoretical or research-focused work.

    For Match Analysis: Conduct a detailed analysis of how the student's resume matches each internship. Provide a brief summary of the match analysis for each opportunity.


    For Output:
    Use the following JSON array format to provide the top-matched internships:


    [
        {
            "name": "Job Title",
            "company": "Company Name",
            "apply_link": "Application Link",
            "match_analysis": "Detailed match analysis here. Explain how the student's background matches the internship requirements, using specific examples."
        },
        {
            "name": "Another Job Title",
            "company": "Another Company",
            "apply_link": "Application Link",
            "match_analysis": "Provide a detailed match analysis for this internship opportunity as well, highlighting relevant matches."
        }
    ]

    If there are no suitable matches, return None. Ensure that no additional words or JSON annotations are included outside the code block.

    """
    
    context = dspy.InputField(desc="Internships")
    resume = dspy.InputField(desc="resume")
    output = dspy.OutputField(desc="list of internships",type=list[JobListing])

class generate_query(dspy.Signature):
    """
    Generate query to search in the weaviate database hybrid search by following below rules:
    1. Analyze the resume, extract keywords from skills, education, experience, projects
    2. then use the keywords to generate query to search in the weaviate database
    3. query should be keyword based to find the best internships for the resume
    """

    context = dspy.InputField(desc="Resume")
    query = dspy.OutputField(desc="query in simple string format")


def main():
    
        
    file = st.file_uploader("Upload Resume to get started", type=["pdf"])
    my_bar.progress(0,text="Starting...") 
    
    if file is not None:
        msg = st.toast("Resume Uploaded")
        if check_resume(file):
            with st.status("Extracting Details from  Resume"):
                resume = resume_into_json(file)
                st.write(resume)

            analysis = Internship_finder()
            
            my_bar.progress(30,text="Finding Internships")   
            
            generate = analysis(resume)
            print(generate)

            if generate != "None":
                st.subheader("List of Internships:")
                col_company, col_url = st.columns([2,6])
                interns = json.loads(generate)
                my_bar.progress(100, "Internships Found !!")
              
                with col_company:
                        for intern in interns:
                            st.link_button(intern["company"],company_url(intern["company"]))
                            
                    
                with col_url:
                        for intern in interns:
                            st.link_button(intern["name"], intern["apply_link"])
                            with st.status("Match Analysis"):
                                st.write(intern["match_analysis"])
                

            else:
                my_bar.progress(100, "Sorry, No Internships Found for you !!")
                st.write(" We are adding more internships every day, please check back later.")
            
            
        else:
            st.warning("Invalid File Uploaded !!")
            my_bar.progress(0,text="Invalid File Uploaded")


if __name__ == "__main__":
    main()
