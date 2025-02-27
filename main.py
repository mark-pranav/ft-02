from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import google.generativeai as genai
import PyPDF2 as pdf
import json
import pandas as pd
import io
import re
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="ATS Resume Analyzer API",
    description="API for analyzing resumes against job descriptions using Gemini AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models
class ATSRequest(BaseModel):
    job_description: str = Field(..., description="The job description to match against")


class MissingKeywords(BaseModel):
    keywords: List[str] = Field(default_factory=list, description="Keywords missing from the resume")


class ResumeResult(BaseModel):
    resume_name: str = Field(..., description="Filename of the resume")
    score: float = Field(..., description="Match score percentage (0-100)")
    missing_keywords: List[str] = Field(default_factory=list, description="Keywords missing from the resume")
    profile_summary: str = Field(..., description="Detailed analysis and suggestions")


class ATSResponse(BaseModel):
    results: List[ResumeResult] = Field(..., description="Analysis results for each resume")


# Dependency for API key
def get_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY not configured on server"
        )
    return api_key


# Configure Gemini API
def configure_genai(api_key: str):
    genai.configure(api_key=api_key)


# Extract text from PDF
def extract_pdf_text(pdf_bytes):
    try:
        reader = pdf.PdfReader(io.BytesIO(pdf_bytes))
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return " ".join(text)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"PDF extraction error: {str(e)}"
        )


# Prepare prompt for Gemini
def prepare_prompt(resume_text, job_description):
    prompt_template = """
    Act as an expert ATS (Applicant Tracking System) specialist with deep expertise in:
    - Technical fields
    - Software engineering
    - Data science
    - Data analysis
    - Big data engineering

    Evaluate the following resume against the job description. Consider that the job market
    is highly competitive. Provide detailed feedback for resume improvement.

    Resume:
    {resume_text}

    Job Description:
    {job_description}

    Provide a response in the following JSON format ONLY:
    {{
        "JD Match": "percentage between 0-100",
        "MissingKeywords": ["keyword1", "keyword2", ...],
        "Profile Summary": "detailed analysis of the match and specific improvement suggestions"
    }}
    """

    return prompt_template.format(
        resume_text=resume_text.strip(),
        job_description=job_description.strip()
    )


# Get response from Gemini API
def get_gemini_response(prompt, api_key):
    try:
        configure_genai(api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(prompt)

        if not response or not response.text:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Empty response received from Gemini API"
            )

        try:
            response_json = json.loads(response.text)

            required_fields = ["JD Match", "MissingKeywords", "Profile Summary"]
            for field in required_fields:
                if field not in response_json:
                    raise ValueError(f"Missing required field: {field}")

            return response_json

        except json.JSONDecodeError:
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, response.text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail="Could not parse extracted JSON content from Gemini response"
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Could not extract valid JSON from Gemini response"
                )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini API error: {str(e)}"
        )


# Process resume file
async def process_resume(file: UploadFile, job_description: str, api_key: str):
    try:
        filename = file.filename
        content = await file.read()

        # Extract text
        resume_text = extract_pdf_text(content)

        # Generate prompt and get response
        prompt = prepare_prompt(resume_text, job_description)
        response = get_gemini_response(prompt, api_key)

        # Extract match percentage
        match_percentage = response["JD Match"]
        match = re.search(r'(\d+)', str(match_percentage))
        score = float(match.group(1)) if match else 0.0

        # Create result
        result = {
            "resume_name": filename,
            "score": score,
            "missing_keywords": response["MissingKeywords"],
            "profile_summary": response["Profile Summary"]
        }

        return result

    except HTTPException as e:
        # Re-raise HTTP exceptions as is
        raise e
    except Exception as e:
        # Wrap other exceptions
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing resume {file.filename}: {str(e)}"
        )


# Endpoints
@app.post("/analyze", response_model=ATSResponse,
          summary="Analyze resumes against a job description",
          description="Upload multiple resume PDFs and a job description to get ATS analysis")
async def analyze_resumes(
        background_tasks: BackgroundTasks,
        job_description: str = Form(...),
        resumes: List[UploadFile] = File(...),
        api_key: str = Depends(get_api_key)
):
    if not resumes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No resume files uploaded"
        )

    results = []
    for resume in resumes:
        if not resume.filename.lower().endswith('.pdf'):
            results.append({
                "resume_name": resume.filename,
                "score": 0.0,
                "missing_keywords": [],
                "profile_summary": "Error: File is not a PDF"
            })
            continue

        try:
            result = await process_resume(resume, job_description, api_key)
            results.append(result)
            # Add small delay between API calls
            if resume != resumes[-1]:  # Don't delay after the last resume
                await asyncio.sleep(1)
        except Exception as e:
            results.append({
                "resume_name": resume.filename,
                "score": 0.0,
                "missing_keywords": [],
                "profile_summary": f"Error processing resume: {str(e)}"
            })

    # Sort results by score
    ranked_results = sorted(results, key=lambda x: x["score"], reverse=True)

    return {"results": ranked_results}


# Health check endpoint
@app.get("/health",
         summary="Health check endpoint",
         description="Check if the API is running")
def health_check():
    return {"status": "healthy"}


# Main driver
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)