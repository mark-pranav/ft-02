from fastapi import FastAPI, Form, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Union, Dict
import google.generativeai as genai
import json
import re
import asyncio
import aiohttp
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(
    title="ATS Resume Analyzer API",
    description="API for analyzing resumes against job descriptions using Gemini AI",
    version="1.3.0"
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
class ResumeInput(BaseModel):
    url: HttpUrl = Field(..., description="URL to the resume image file (JPG format)")
    filename: Optional[str] = Field(None, description="Optional custom filename")
    user_id: str = Field(..., description="User ID associated with this resume")


class ATSRequest(BaseModel):
    job_description: str = Field(..., description="The job description to match against")
    resumes: List[ResumeInput] = Field(..., description="List of resume URLs to analyze with user IDs")


class UserResult(BaseModel):
    user_id: str = Field(..., description="User ID")
    score: float = Field(..., description="Match score percentage (0-100)")
    content_score: float = Field(..., description="Content match score (0-50)")
    keyword_score: float = Field(..., description="Keyword match score (0-50)")
    missing_keywords: List[str] = Field(..., description="Missing keywords from the resume")
    sgAnalysis: str = Field(..., description="Analysis of skills gap between resume and job requirements")


class ATSResponse(BaseModel):
    results: List[UserResult] = Field(..., description="Analysis results for each user")


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


# Define technology/skill variations mapping
# This helps in recognizing different variations of the same technology
SKILL_VARIATIONS = {
    # JavaScript frameworks/libraries
    "react": ["reactjs", "react.js", "react js"],
    "angular": ["angularjs", "angular.js", "angular js"],
    "vue": ["vuejs", "vue.js", "vue js"],
    "node": ["nodejs", "node.js", "node js"],

    # Programming languages
    "javascript": ["js", "ecmascript"],
    "typescript": ["ts"],
    "python": ["py"],
    "java": ["jdk"],
    "c#": ["csharp", "c sharp"],
    "c++": ["cpp", "cplusplus", "c plus plus"],

    # Database technologies
    "postgresql": ["postgres", "pgsql"],
    "mongodb": ["mongo"],
    "mysql": ["sql"],
    "mssql": ["sql server", "microsoft sql server"],

    # Cloud platforms
    "aws": ["amazon web services"],
    "gcp": ["google cloud platform", "google cloud"],
    "azure": ["microsoft azure"],

    # Big data technologies
    "hadoop": ["apache hadoop"],
    "spark": ["apache spark"],
    "kafka": ["apache kafka"],

    # DevOps tools
    "docker": ["containerization"],
    "kubernetes": ["k8s"],
    "jenkins": ["ci/cd", "cicd"],
    "terraform": ["infrastructure as code", "iac"],

    # Mobile development
    "react native": ["reactnative"],
    "flutter": ["dart flutter"],
    "swift": ["ios development"],
    "kotlin": ["android development"],

    # AI/ML
    "tensorflow": ["tf"],
    "pytorch": ["torch"],
    "machine learning": ["ml"],
    "deep learning": ["dl"],

    # Others
    "restful api": ["rest api", "rest", "restful"],
    "graphql": ["gql"],
    "html5": ["html"],
    "css3": ["css"],
    "sass": ["scss"],
    "less": ["css preprocessor"],
}

# Create reverse mapping for easier lookup
REVERSE_SKILL_MAP = {}
for main_skill, variations in SKILL_VARIATIONS.items():
    for variant in variations:
        REVERSE_SKILL_MAP[variant] = main_skill
    # Also map the main skill to itself
    REVERSE_SKILL_MAP[main_skill] = main_skill


# Normalize a keyword to its canonical form
def normalize_keyword(keyword):
    # Clean the keyword: strip whitespace, make lowercase
    cleaned = keyword.strip().lower()

    # Remove common suffixes
    for suffix in [".js", " js", "-js"]:
        if cleaned.endswith(suffix):
            potential_base = cleaned[:-len(suffix)]
            # Only remove suffix if the base exists in our map
            if potential_base in REVERSE_SKILL_MAP:
                cleaned = potential_base
                break

    # Check if this is a known variation
    if cleaned in REVERSE_SKILL_MAP:
        return REVERSE_SKILL_MAP[cleaned]

    return cleaned

# Prepare prompt for Gemini (for direct image processing)
def prepare_image_prompt(resume_url, job_description):
    prompt_template = """
    You are looking at a resume image. First, extract all the text content from the resume.

    Then act as an expert ATS (Applicant Tracking System) specialist with deep expertise in Technical fields like:

    - Software engineering
    - Data science
    - Data analysis
    - Big data engineering
    - Frontend Developer
    - Backend Developer
    - DevOps Engineer
    - Programming Specialist

    Evaluate the resume against the job description using a balanced scoring system:
    - ContentMatch (0-50 points): Evaluate how well the candidate's experience, education, and overall profile aligns with the job requirements
    - KeywordMatch (0-50 points): Evaluate how many of the specific keywords, technologies, and skills from the job description appear in the resume

    IMPORTANT: When matching keywords, skills, and technologies, be intelligent about variations:
    - Consider "React", "ReactJS", and "React.js" as the same technology
    - Recognize when technologies are mentioned with slight variations (like "Node.js" vs "Node")
    - Match skill abbreviations with their full names (like "ML" with "Machine Learning")
    - Don't penalize for these variations - they should count as matches, not missing keywords

    Consider that the job market is highly competitive. Provide detailed feedback for resume improvement.

    Job Description:
    {job_description}

    Provide a response in the following JSON format ONLY, with no additional text:
    {{
        "ContentMatch": "Score (0-50) for overall content/experience alignment with job description",
        "KeywordMatch": "Score (0-50) for matching of specific keywords and skills",
        "TotalScore": "Sum of ContentMatch and KeywordMatch (0-100)",
        "MissingKeywords": ["keyword1", "keyword2", ...],
        "Profile Summary": "A concise 3-sentence evaluation highlighting strengths, key gaps, and actionable improvement suggestions.",
        "Skills Gap Analysis": "comprehensive analysis of the specific skills gap between the candidate's resume and the job requirements, including technical skills, tools, methodologies, and experience levels that are missing or insufficient"
    }}

    IMPORTANT: The field "Skills Gap Analysis" MUST be exactly 3 sentences long, no more and no less, and must use exactly that field name with spaces.
    """

    return prompt_template.format(
        job_description=job_description.strip()
    )
# Prepare prompt for Gemini (for direct image processing)
# def prepare_image_prompt(resume_url, job_description):
#     prompt_template = """
#     You are looking at a resume image. First, extract all the text content from the resume.
#
#     Then act as an expert ATS (Applicant Tracking System) specialist with deep expertise in Technical fields like:
#
#     - Software engineering
#     - Data science
#     - Data analysis
#     - Big data engineering
#     - Frontend Developer
#     - Backend Developer
#     - DevOps Engineer
#     - Programming Specialist
#
#     Evaluate the resume against the job description using a balanced scoring system:
#     - ContentMatch (0-50 points): Evaluate how well the candidate's experience, education, and overall profile aligns with the job requirements
#     - KeywordMatch (0-50 points): Evaluate how many of the specific keywords, technologies, and skills from the job description appear in the resume
#
#     IMPORTANT: When matching keywords, skills, and technologies, be intelligent about variations:
#     - Consider "React", "ReactJS", and "React.js" as the same technology
#     - Recognize when technologies are mentioned with slight variations (like "Node.js" vs "Node")
#     - Match skill abbreviations with their full names (like "ML" with "Machine Learning")
#     - Don't penalize for these variations - they should count as matches, not missing keywords
#
#     Consider that the job market is highly competitive. Provide detailed feedback for resume improvement.
#
#     Job Description:
#     {job_description}
#
#     Provide a response in the following JSON format ONLY, with no additional text:
#     {{
#         "ContentMatch": "Score (0-50) for overall content/experience alignment with job description",
#         "KeywordMatch": "Score (0-50) for matching of specific keywords and skills",
#         "TotalScore": "Sum of ContentMatch and KeywordMatch (0-100)",
#         "MissingKeywords": ["keyword1", "keyword2", ...],
#         "Profile Summary": "A concise 3-sentence evaluation highlighting strengths, key gaps, and actionable improvement suggestions.",
#         "Skills Gap Analysis": "comprehensive analysis of the specific skills gap between the candidate's resume and the job requirements, including technical skills, tools, methodologies, and experience levels that are missing or insufficient"
#     }}
#     """
#
#     return prompt_template.format(
#         job_description=job_description.strip()
#     )


# Get response from Gemini API with image URL
# Update the get_gemini_image_response function to handle different possible field formats
async def get_gemini_image_response(prompt, image_url, api_key, max_retries=3, retry_delay=2):
    configure_genai(api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    for attempt in range(max_retries):
        try:
            response = model.generate_content([prompt, image_url])
            logger.info(f"Received response from Gemini Vision API (attempt {attempt + 1})")

            if not response or not response.text:
                logger.warning(f"Empty response from Gemini API (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail="Empty response received from Gemini API after retries"
                    )

            try:
                # Log the raw response for debugging
                logger.info(f"Raw Gemini response: {response.text[:500]}...")

                # Try to parse the JSON response
                json_pattern = r'\{.*\}'
                match = re.search(json_pattern, response.text, re.DOTALL)
                if match:
                    try:
                        extracted_json = match.group()
                        logger.info(f"Extracted JSON: {extracted_json[:500]}...")
                        response_json = json.loads(extracted_json)

                        # Check for required fields - ignoring case and alternative formats
                        required_fields = ["ContentMatch", "KeywordMatch", "TotalScore", "MissingKeywords"]
                        for field in required_fields:
                            if field not in response_json and field.lower() not in {k.lower() for k in
                                                                                    response_json.keys()}:
                                logger.warning(f"Missing required field in Gemini response: {field}")
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(retry_delay)
                                    continue
                                else:
                                    raise ValueError(f"Missing required field: {field}")

                        # Ensure Skills Gap Analysis field exists in some form
                        has_gap_analysis = False
                        for key in response_json.keys():
                            if key.lower().replace(" ", "").replace("_", "") == "skillsgapanalysis":
                                has_gap_analysis = True
                                break

                        if not has_gap_analysis:
                            logger.warning("Missing Skills Gap Analysis field in response")
                            # Add a default value if missing
                            response_json["Skills Gap Analysis"] = "Skills gap analysis not available"

                        return response_json
                    except json.JSONDecodeError:
                        if attempt < max_retries - 1:
                            logger.warning("JSON decode error in extracted text, retrying...")
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            raise HTTPException(
                                status_code=status.HTTP_502_BAD_GATEWAY,
                                detail="Could not parse JSON content from Gemini response"
                            )
                else:
                    if attempt < max_retries - 1:
                        logger.warning("Could not extract JSON pattern, retrying...")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_502_BAD_GATEWAY,
                            detail="Could not extract valid JSON from Gemini response"
                        )

            except Exception as e:
                logger.error(f"Error processing Gemini response: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=f"Error processing Gemini response after retries: {str(e)}"
                    )

        except Exception as e:
            logger.error(f"Error with Gemini API: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                continue
            else:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Gemini API error after retries: {str(e)}"
                )


# Extract score from various formats
def extract_score_value(match_percentage):
    logger.info(f"Extracting score value from: {match_percentage}")

    # Handle if it's already a number
    if isinstance(match_percentage, (int, float)):
        return float(match_percentage)

    # Handle if it's a string containing just a number
    if isinstance(match_percentage, str):
        try:
            # Try direct conversion first
            return float(match_percentage.strip())
        except ValueError:
            # If that fails, try to extract a number with regex
            pass

    # Use regex to find a number pattern in various formats
    patterns = [
        r'(\d+\.?\d*)',  # Match numbers like 75 or 75.5
        r'(\d+\.?\d*)%',  # Match percentage format like 75% or 75.5%
        r'(\d+\.?\d*)\s*percent',  # Match "75 percent" or "75.5 percent"
        r'(\d+\.?\d*)\s*out of\s*100',  # Match "75 out of 100"
        r'(\d+\.?\d*)\s*out of\s*50',  # Match "35 out of 50"
    ]

    for pattern in patterns:
        match = re.search(pattern, str(match_percentage))
        if match:
            try:
                extracted_value = float(match.group(1))
                logger.info(f"Successfully extracted score value: {extracted_value}")
                return extracted_value
            except ValueError:
                continue

    # If all extraction methods fail, log and return 0
    logger.warning(f"Could not extract score value from: {match_percentage}, defaulting to 0")
    return 0.0


# Process missing keywords to normalize variations
def process_missing_keywords(missing_keywords):
    if isinstance(missing_keywords, str):
        # Sometimes the API might return a string instead of a list
        try:
            missing_keywords = json.loads(missing_keywords)
        except json.JSONDecodeError:
            # If string can't be parsed as JSON, convert to list by splitting
            missing_keywords = [kw.strip() for kw in missing_keywords.split(',')]

    # Normalize each keyword
    normalized_keywords = []
    for keyword in missing_keywords:
        normalized = normalize_keyword(keyword)
        # Only add if not already in the list (avoid duplicates)
        if normalized and normalized not in normalized_keywords:
            normalized_keywords.append(normalized)

    return normalized_keywords


# Extract scores from response
def extract_scores(response):
    logger.info(f"Extracting scores from response")

    # Get the component scores
    content_score = extract_score_value(response.get("ContentMatch", "0"))
    keyword_score = extract_score_value(response.get("KeywordMatch", "0"))

    # Get the total score directly or calculate it
    total_score = extract_score_value(response.get("TotalScore", "0"))
    calculated_total = content_score + keyword_score

    # If the extracted total seems wrong, use the calculated total
    if abs(total_score - calculated_total) > 5:  # Allow small differences due to rounding
        logger.warning(
            f"Total score ({total_score}) doesn't match sum of components ({calculated_total}), using calculated total")
        total_score = calculated_total

    # Ensure scores are in proper ranges
    content_score = max(0, min(50, content_score))
    keyword_score = max(0, min(50, keyword_score))
    total_score = max(0, min(100, total_score))

    return {
        "content_score": content_score,
        "keyword_score": keyword_score,
        "total_score": total_score
    }


async def process_resume_url(resume_input: ResumeInput, job_description: str, api_key: str):
    try:
        url = str(resume_input.url)
        user_id = resume_input.user_id

        logger.info(f"Processing resume for user: {user_id} from URL: {url}")

        # Generate prompt for image analysis
        prompt = prepare_image_prompt(url, job_description)

        # Send image URL directly to Gemini Vision API
        response = await get_gemini_image_response(prompt, url, api_key)
        logger.info(f"Received analysis for user: {user_id}")

        # Extract scores
        scores = extract_scores(response)

        # Extract and process missing keywords
        missing_keywords = process_missing_keywords(response.get("MissingKeywords", []))

        # Look for Skills Gap Analysis with different possible formats
        skills_gap_analysis = None
        for key in response.keys():
            if key.lower().replace(" ", "").replace("_", "") == "skillsgapanalysis":
                skills_gap_analysis = response[key]
                break

        if skills_gap_analysis is None:
            skills_gap_analysis = "Skills gap analysis not available"

        # Create result with user_id
        result = {
            "user_id": user_id,
            "score": scores["total_score"],
            "content_score": scores["content_score"],
            "keyword_score": scores["keyword_score"],
            "missing_keywords": missing_keywords,
            "sgAnalysis": skills_gap_analysis
        }

        logger.info(f"Analysis complete for user: {user_id}, total score: {scores['total_score']}")
        return result

    except HTTPException as e:
        # Re-raise HTTP exceptions as is
        logger.error(f"HTTP exception while processing resume for user {user_id}: {str(e)}")
        raise e
    except Exception as e:
        # Wrap other exceptions
        logger.error(f"Error processing resume for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing resume for user {user_id}: {str(e)}"
        )
# Process resume from URL directly with Gemini Vision
# async def process_resume_url(resume_input: ResumeInput, job_description: str, api_key: str):
#     try:
#         url = str(resume_input.url)
#         user_id = resume_input.user_id
#
#         logger.info(f"Processing resume for user: {user_id} from URL: {url}")
#
#         # Generate prompt for image analysis
#         prompt = prepare_image_prompt(url, job_description)
#
#         # Send image URL directly to Gemini Vision API
#         response = await get_gemini_image_response(prompt, url, api_key)
#         logger.info(f"Received analysis for user: {user_id}")
#
#         # Extract scores
#         scores = extract_scores(response)
#
#         # Extract and process missing keywords
#         missing_keywords = process_missing_keywords(response.get("MissingKeywords", []))
#
#         # Create result with user_id
#         result = {
#             "user_id": user_id,
#             "score": scores["total_score"],
#             "content_score": scores["content_score"],
#             "keyword_score": scores["keyword_score"],
#             "missing_keywords": missing_keywords,
#             "sgAnalysis": response.get("Skills Gap Analysis", "Skills gap analysis not available")
#         }
#
#         logger.info(f"Analysis complete for user: {user_id}, total score: {scores['total_score']}")
#         return result
#
#     except HTTPException as e:
#         # Re-raise HTTP exceptions as is
#         logger.error(f"HTTP exception while processing resume for user {user_id}: {str(e)}")
#         raise e
#     except Exception as e:
#         # Wrap other exceptions
#         logger.error(f"Error processing resume for user {user_id}: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error processing resume for user {user_id}: {str(e)}"
#         )


# Process multiple resumes in parallel with rate limiting
async def process_resumes(resumes: List[ResumeInput], job_description: str, api_key: str):
    # Use a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(3)  # Process up to 3 resumes concurrently

    async def process_with_semaphore(resume):
        async with semaphore:
            try:
                return await process_resume_url(resume, job_description, api_key)
            except Exception as e:
                logger.error(f"Error in process_with_semaphore for user {resume.user_id}: {str(e)}")
                return {
                    "user_id": resume.user_id,
                    "score": 0.0,
                    "content_score": 0.0,
                    "keyword_score": 0.0,
                    "missing_keywords": [],
                    "sgAnalysis": "Could not analyze skills gap due to processing error."
                }

    # Create tasks for all resumes
    tasks = [process_with_semaphore(resume) for resume in resumes]

    # Execute all tasks
    results = await asyncio.gather(*tasks)

    # Sort results by score
    ranked_results = sorted(results, key=lambda x: x["score"], reverse=True)

    return ranked_results


# Endpoints
@app.post("/analyze", response_model=ATSResponse,
          summary="Analyze resumes against a job description",
          description="Provide resume image URLs with user IDs and a job description to get ATS analysis")
async def analyze_resumes_endpoint(
        request: ATSRequest,
        api_key: str = Depends(get_api_key)
):
    if not request.resumes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No resume URLs provided"
        )

    logger.info(f"Received analyze request with {len(request.resumes)} resumes")
    results = await process_resumes(request.resumes, request.job_description, api_key)
    logger.info(f"Analysis complete for {len(results)} resumes")
    return {"results": results}


# Form-based endpoint for backward compatibility
@app.post("/analyze-form", response_model=ATSResponse,
          summary="Analyze resumes against a job description (form-based)",
          description="Provide resume URLs with user IDs and a job description using form data")
async def analyze_resumes_form(
        job_description: str = Form(...),
        resume_urls: str = Form(...),  # Comma-separated URLs
        user_ids: str = Form(...),  # Comma-separated user IDs
        api_key: str = Depends(get_api_key)
):
    # Parse resume URLs and user IDs from form data
    urls = [url.strip() for url in resume_urls.split(",") if url.strip()]
    ids = [uid.strip() for uid in user_ids.split(",") if uid.strip()]

    if not urls:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No resume URLs provided"
        )

    if len(urls) != len(ids):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The number of resume URLs must match the number of user IDs"
        )

    logger.info(f"Received form-based analyze request with {len(urls)} resumes")

    # Convert to ResumeInput objects with user IDs
    resume_inputs = [ResumeInput(url=url, user_id=uid) for url, uid in zip(urls, ids)]

    results = await process_resumes(resume_inputs, job_description, api_key)
    logger.info(f"Form-based analysis complete for {len(results)} resumes")
    return {"results": results}


# Health check endpoint
@app.get("/health",
         summary="Health check endpoint",
         description="Check if the API is running")
def health_check():
    return {"status": "healthy"}


# Main driver
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting ATS Resume Analyzer API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI, Form, HTTPException, Depends, status, BackgroundTasks
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel, Field, HttpUrl
# from typing import List, Optional, Union
# import google.generativeai as genai
# import json
# import re
# import asyncio
# import aiohttp
# import os
# import logging
# from dotenv import load_dotenv
#
# # Configure logging
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
# # Load environment variables
# load_dotenv()
#
# app = FastAPI(
#     title="ATS Resume Analyzer API",
#     description="API for analyzing resumes against job descriptions using Gemini AI",
#     version="1.2.0"
# )
#
# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Modify in production to specific origins
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
#
# # Models
# class ResumeInput(BaseModel):
#     url: HttpUrl = Field(..., description="URL to the resume image file (JPG format)")
#     filename: Optional[str] = Field(None, description="Optional custom filename")
#     user_id: str = Field(..., description="User ID associated with this resume")
#
#
# class ATSRequest(BaseModel):
#     job_description: str = Field(..., description="The job description to match against")
#     resumes: List[ResumeInput] = Field(..., description="List of resume URLs to analyze with user IDs")
#
#
# class UserResult(BaseModel):
#     user_id: str = Field(..., description="User ID")
#     score: float = Field(..., description="Match score percentage (0-100)")
#     sgAnalysis: str = Field(..., description="Analysis of skills gap between resume and job requirements")
#
#
# class ATSResponse(BaseModel):
#     results: List[UserResult] = Field(..., description="Analysis results for each user")
#
#
# # Dependency for API key
# def get_api_key():
#     api_key = os.getenv("GEMINI_API_KEY")
#     if not api_key:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="GEMINI_API_KEY not configured on server"
#         )
#     return api_key
#
#
# # Configure Gemini API
# def configure_genai(api_key: str):
#     genai.configure(api_key=api_key)
#
#
# # Prepare prompt for Gemini (for direct image processing)
# def prepare_image_prompt(resume_url, job_description):
#     prompt_template = """
#     You are looking at a resume image. First, extract all the text content from the resume.
#
#     Then act as an expert ATS (Applicant Tracking System) specialist with deep expertise in Technical fields like:
#
#     - Software engineering
#     - Data science
#     - Data analysis
#     - Big data engineering
#     - Frontend Developer
#     - Backend Developer
#     - DevOps Engineer
#     - Programmming Specialist
#
#     Evaluate the resume against the job description. Consider that the job market
#     is highly competitive. Provide detailed feedback for resume improvement.
#
#     Job Description:
#     {job_description}
#
#     Provide a response in the following JSON format ONLY, with no additional text:
#     {{
#         "ContentMatch": "Score (0-50) for overall content/experience alignment with job description",
#         "KeywordMatch": "Score (0-50) for matching of specific keywords and skills",
#         "TotalScore": "Sum of ContentMatch and KeywordMatch (0-100)",
#         "MissingKeywords": ["keyword1", "keyword2", ...],
#         "Profile Summary": "A concise 3-sentence evaluation highlighting strengths, key gaps, and actionable improvement suggestions.",
#         "Skills Gap Analysis": "comprehensive analysis of the specific skills gap between the candidate's resume and the job requirements, including technical skills, tools, methodologies, and experience levels that are missing or insufficient"
#     }}
#     """
#     return prompt_template.format(job_description=job_description.strip())
# # def prepare_image_prompt(resume_url, job_description):
# #     prompt_template = """
# #     You are looking at a resume image. First, extract all the text content from the resume.
# #
# #     Then act as an expert ATS (Applicant Tracking System) specialist with deep expertise in Technical fields like:
# #
# #     - Software engineering
# #     - Data science
# #     - Data analysis
# #     - Big data engineering
# #     - Frontend Developer
# #     - Backend Developer
# #     - DevOps Engineer
# #     - Programmming Specialist
# #
# #     Evaluate the resume against the job description. Consider that the job market
# #     is highly competitive. Provide detailed feedback for resume improvement.
# #
# #     Job Description:
# #     {job_description}
# #
# #     Provide a response in the following JSON format ONLY, with no additional text:
# #     {{
# #         "JD Match": "Percentage (0-100) indicating alignment with the job description",
# #         "MissingKeywords": ["keyword1", "keyword2", ...],
# #         "Profile Summary": "A concise 3-sentence evaluation highlighting strengths, key gaps, and actionable improvement suggestions.",
# #         "Skills Gap Analysis": "comprehensive analysis of the specific skills gap between the candidate's resume and the job requirements, including technical skills, tools, methodologies, and experience levels that are missing or insufficient"
# #     }}
# #     """
# #
# #     return prompt_template.format(
# #         job_description=job_description.strip()
# #     )
#
#
# # Get response from Gemini API with image URL
# async def get_gemini_image_response(prompt, image_url, api_key, max_retries=3, retry_delay=2):
#     configure_genai(api_key)
#     model = genai.GenerativeModel('gemini-1.5-flash')
#
#     for attempt in range(max_retries):
#         try:
#             response = model.generate_content([prompt, image_url])
#             logger.info(f"Received response from Gemini Vision API (attempt {attempt + 1})")
#
#             if not response or not response.text:
#                 logger.warning(f"Empty response from Gemini API (attempt {attempt + 1})")
#                 if attempt < max_retries - 1:
#                     await asyncio.sleep(retry_delay)
#                     continue
#                 else:
#                     raise HTTPException(
#                         status_code=status.HTTP_502_BAD_GATEWAY,
#                         detail="Empty response received from Gemini API after retries"
#                     )
#
#             try:
#                 # Log the raw response for debugging
#                 logger.info(f"Raw Gemini response: {response.text[:500]}...")
#
#                 # Try to parse the JSON response
#                 json_pattern = r'\{.*\}'
#                 match = re.search(json_pattern, response.text, re.DOTALL)
#                 if match:
#                     try:
#                         extracted_json = match.group()
#                         logger.info(f"Extracted JSON: {extracted_json[:500]}...")
#                         response_json = json.loads(extracted_json)
#
#                         # Check for required fields
#                         required_fields = ["JD Match", "MissingKeywords", "Profile Summary", "Skills Gap Analysis"]
#                         for field in required_fields:
#                             if field not in response_json:
#                                 logger.warning(f"Missing required field in Gemini response: {field}")
#                                 if attempt < max_retries - 1:
#                                     await asyncio.sleep(retry_delay)
#                                     continue
#                                 else:
#                                     raise ValueError(f"Missing required field: {field}")
#
#                         return response_json
#                     except json.JSONDecodeError:
#                         if attempt < max_retries - 1:
#                             logger.warning("JSON decode error in extracted text, retrying...")
#                             await asyncio.sleep(retry_delay)
#                             continue
#                         else:
#                             raise HTTPException(
#                                 status_code=status.HTTP_502_BAD_GATEWAY,
#                                 detail="Could not parse JSON content from Gemini response"
#                             )
#                 else:
#                     if attempt < max_retries - 1:
#                         logger.warning("Could not extract JSON pattern, retrying...")
#                         await asyncio.sleep(retry_delay)
#                         continue
#                     else:
#                         raise HTTPException(
#                             status_code=status.HTTP_502_BAD_GATEWAY,
#                             detail="Could not extract valid JSON from Gemini response"
#                         )
#
#             except Exception as e:
#                 logger.error(f"Error processing Gemini response: {str(e)}")
#                 if attempt < max_retries - 1:
#                     await asyncio.sleep(retry_delay)
#                     continue
#                 else:
#                     raise HTTPException(
#                         status_code=status.HTTP_502_BAD_GATEWAY,
#                         detail=f"Error processing Gemini response after retries: {str(e)}"
#                     )
#
#         except Exception as e:
#             logger.error(f"Error with Gemini API: {str(e)}")
#             if attempt < max_retries - 1:
#                 await asyncio.sleep(retry_delay)
#                 continue
#             else:
#                 raise HTTPException(
#                     status_code=status.HTTP_502_BAD_GATEWAY,
#                     detail=f"Gemini API error after retries: {str(e)}"
#                 )
#
#
# # Extract score from various formats
# def extract_score(match_percentage):
#     logger.info(f"Extracting score from: {match_percentage}")
#
#     # Handle if it's already a number
#     if isinstance(match_percentage, (int, float)):
#         return float(match_percentage)
#
#     # Handle if it's a string containing just a number
#     if isinstance(match_percentage, str):
#         try:
#             # Try direct conversion first
#             return float(match_percentage.strip())
#         except ValueError:
#             # If that fails, try to extract a number with regex
#             pass
#
#     # Use regex to find a number pattern in various formats
#     patterns = [
#         r'(\d+\.?\d*)',  # Match numbers like 75 or 75.5
#         r'(\d+\.?\d*)%',  # Match percentage format like 75% or 75.5%
#         r'(\d+\.?\d*)\s*percent',  # Match "75 percent" or "75.5 percent"
#         r'(\d+\.?\d*)\s*out of\s*100',  # Match "75 out of 100"
#     ]
#
#     for pattern in patterns:
#         match = re.search(pattern, str(match_percentage))
#         if match:
#             try:
#                 extracted_value = float(match.group(1))
#                 logger.info(f"Successfully extracted score: {extracted_value}")
#                 return extracted_value
#             except ValueError:
#                 continue
#
#     # If all extraction methods fail, log and return 0
#     logger.warning(f"Could not extract score from: {match_percentage}, defaulting to 0")
#     return 0.0
#
#
# # Process resume from URL directly with Gemini Vision
# async def process_resume_url(resume_input: ResumeInput, job_description: str, api_key: str):
#     try:
#         url = str(resume_input.url)
#         user_id = resume_input.user_id
#
#         logger.info(f"Processing resume for user: {user_id} from URL: {url}")
#
#         # Generate prompt for image analysis
#         prompt = prepare_image_prompt(url, job_description)
#
#         # Send image URL directly to Gemini Vision API
#         response = await get_gemini_image_response(prompt, url, api_key)
#         logger.info(f"Received analysis for user: {user_id}")
#
#         # Extract match percentage - with improved handling
#         match_percentage = response.get("JD Match", "0")
#         score = extract_score(match_percentage)
#
#         # Create result with user_id
#         result = {
#             "user_id": user_id,
#             "score": score,
#             "sgAnalysis": response.get("Skills Gap Analysis", "Skills gap analysis not available")
#         }
#
#         logger.info(f"Analysis complete for user: {user_id}, score: {score}")
#         return result
#
#     except HTTPException as e:
#         # Re-raise HTTP exceptions as is
#         logger.error(f"HTTP exception while processing resume for user {user_id}: {str(e)}")
#         raise e
#     except Exception as e:
#         # Wrap other exceptions
#         logger.error(f"Error processing resume for user {user_id}: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error processing resume for user {user_id}: {str(e)}"
#         )
#
#
# # Process multiple resumes in parallel with rate limiting
# async def process_resumes(resumes: List[ResumeInput], job_description: str, api_key: str):
#     # Use a semaphore to limit concurrent requests
#     semaphore = asyncio.Semaphore(3)  # Process up to 3 resumes concurrently
#
#     async def process_with_semaphore(resume):
#         async with semaphore:
#             try:
#                 return await process_resume_url(resume, job_description, api_key)
#             except Exception as e:
#                 logger.error(f"Error in process_with_semaphore for user {resume.user_id}: {str(e)}")
#                 return {
#                     "user_id": resume.user_id,
#                     "score": 0.0,
#                     "sgAnalysis": "Could not analyze skills gap due to processing error."
#                 }
#
#     # Create tasks for all resumes
#     tasks = [process_with_semaphore(resume) for resume in resumes]
#
#     # Execute all tasks
#     results = await asyncio.gather(*tasks)
#
#     # Sort results by score
#     ranked_results = sorted(results, key=lambda x: x["score"], reverse=True)
#
#     return ranked_results
#
#
# # Endpoints
# @app.post("/analyze", response_model=ATSResponse,
#           summary="Analyze resumes against a job description",
#           description="Provide resume image URLs with user IDs and a job description to get ATS analysis")
# async def analyze_resumes_endpoint(
#         request: ATSRequest,
#         api_key: str = Depends(get_api_key)
# ):
#     if not request.resumes:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="No resume URLs provided"
#         )
#
#     logger.info(f"Received analyze request with {len(request.resumes)} resumes")
#     results = await process_resumes(request.resumes, request.job_description, api_key)
#     logger.info(f"Analysis complete for {len(results)} resumes")
#     return {"results": results}
#
#
# # Form-based endpoint for backward compatibility
# @app.post("/analyze-form", response_model=ATSResponse,
#           summary="Analyze resumes against a job description (form-based)",
#           description="Provide resume URLs with user IDs and a job description using form data")
# async def analyze_resumes_form(
#         job_description: str = Form(...),
#         resume_urls: str = Form(...),  # Comma-separated URLs
#         user_ids: str = Form(...),  # Comma-separated user IDs
#         api_key: str = Depends(get_api_key)
# ):
#     # Parse resume URLs and user IDs from form data
#     urls = [url.strip() for url in resume_urls.split(",") if url.strip()]
#     ids = [uid.strip() for uid in user_ids.split(",") if uid.strip()]
#
#     if not urls:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="No resume URLs provided"
#         )
#
#     if len(urls) != len(ids):
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="The number of resume URLs must match the number of user IDs"
#         )
#
#     logger.info(f"Received form-based analyze request with {len(urls)} resumes")
#
#     # Convert to ResumeInput objects with user IDs
#     resume_inputs = [ResumeInput(url=url, user_id=uid) for url, uid in zip(urls, ids)]
#
#     results = await process_resumes(resume_inputs, job_description, api_key)
#     logger.info(f"Form-based analysis complete for {len(results)} resumes")
#     return {"results": results}
#
#
# # Health check endpoint
# @app.get("/health",
#          summary="Health check endpoint",
#          description="Check if the API is running")
# def health_check():
#     return {"status": "healthy"}
#
#
# # Main driver
# if __name__ == "__main__":
#     import uvicorn
#
#     logger.info("Starting ATS Resume Analyzer API server")
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# from fastapi import FastAPI, Form, HTTPException, Depends, status, BackgroundTasks
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel, Field, HttpUrl
# from typing import List, Optional, Union
# import google.generativeai as genai
# import json
# import re
# import asyncio
# import aiohttp
# import os
# import logging
# from dotenv import load_dotenv
#
# # Configure logging
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
# # Load environment variables
# load_dotenv()
#
# app = FastAPI(
#     title="ATS Resume Analyzer API",
#     description="API for analyzing resumes against job descriptions using Gemini AI",
#     version="1.3.0"
# )
#
# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Modify in production to specific origins
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
#
# # Models
# class ResumeInput(BaseModel):
#     url: HttpUrl = Field(..., description="URL to the resume image file (JPG format)")
#     filename: Optional[str] = Field(None, description="Optional custom filename")
#     user_id: str = Field(..., description="User ID associated with this resume")
#
#
# class ATSRequest(BaseModel):
#     job_description: str = Field(..., description="The job description to match against")
#     resumes: List[ResumeInput] = Field(..., description="List of resume URLs to analyze with user IDs")
#
#
# class UserResult(BaseModel):
#     user_id: str = Field(..., description="User ID")
#     score: float = Field(..., description="Match score percentage (0-100)")
#     content_score: float = Field(..., description="Content match score (0-50)")
#     keyword_score: float = Field(..., description="Keyword match score (0-50)")
#     missing_keywords: List[str] = Field(..., description="Missing keywords from the resume")
#     sgAnalysis: str = Field(..., description="Analysis of skills gap between resume and job requirements")
#
#
# class ATSResponse(BaseModel):
#     results: List[UserResult] = Field(..., description="Analysis results for each user")
#
#
# # Dependency for API key
# def get_api_key():
#     api_key = os.getenv("GEMINI_API_KEY")
#     if not api_key:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="GEMINI_API_KEY not configured on server"
#         )
#     return api_key
#
#
# # Configure Gemini API
# def configure_genai(api_key: str):
#     genai.configure(api_key=api_key)
#
#
# # Prepare prompt for Gemini (for direct image processing)
# def prepare_image_prompt(resume_url, job_description):
#     prompt_template = """
#     You are looking at a resume image. First, extract all the text content from the resume.
#
#     Then act as an expert ATS (Applicant Tracking System) specialist with deep expertise in Technical fields like:
#
#     - Software engineering
#     - Data science
#     - Data analysis
#     - Big data engineering
#     - Frontend Developer
#     - Backend Developer
#     - DevOps Engineer
#     - Programming Specialist
#
#     Evaluate the resume against the job description using a balanced scoring system:
#     - ContentMatch (0-50 points): Evaluate how well the candidate's experience, education, and overall profile aligns with the job requirements
#     - KeywordMatch (0-50 points): Evaluate how many of the specific keywords, technologies, and skills from the job description appear in the resume
#
#     Consider that the job market is highly competitive. Provide detailed feedback for resume improvement.
#
#     Job Description:
#     {job_description}
#
#     Provide a response in the following JSON format ONLY, with no additional text:
#     {{
#         "ContentMatch": "Score (0-50) for overall content/experience alignment with job description",
#         "KeywordMatch": "Score (0-50) for matching of specific keywords and skills",
#         "TotalScore": "Sum of ContentMatch and KeywordMatch (0-100)",
#         "MissingKeywords": ["keyword1", "keyword2", ...],
#         "Profile Summary": "A concise 3-sentence evaluation highlighting strengths, key gaps, and actionable improvement suggestions.",
#         "Skills Gap Analysis": "comprehensive analysis of the specific skills gap between the candidate's resume and the job requirements, including technical skills, tools, methodologies, and experience levels that are missing or insufficient"
#     }}
#     """
#
#     return prompt_template.format(
#         job_description=job_description.strip()
#     )
#
#
# # Get response from Gemini API with image URL
# async def get_gemini_image_response(prompt, image_url, api_key, max_retries=3, retry_delay=2):
#     configure_genai(api_key)
#     model = genai.GenerativeModel('gemini-1.5-flash')
#
#     for attempt in range(max_retries):
#         try:
#             response = model.generate_content([prompt, image_url])
#             logger.info(f"Received response from Gemini Vision API (attempt {attempt + 1})")
#
#             if not response or not response.text:
#                 logger.warning(f"Empty response from Gemini API (attempt {attempt + 1})")
#                 if attempt < max_retries - 1:
#                     await asyncio.sleep(retry_delay)
#                     continue
#                 else:
#                     raise HTTPException(
#                         status_code=status.HTTP_502_BAD_GATEWAY,
#                         detail="Empty response received from Gemini API after retries"
#                     )
#
#             try:
#                 # Log the raw response for debugging
#                 logger.info(f"Raw Gemini response: {response.text[:500]}...")
#
#                 # Try to parse the JSON response
#                 json_pattern = r'\{.*\}'
#                 match = re.search(json_pattern, response.text, re.DOTALL)
#                 if match:
#                     try:
#                         extracted_json = match.group()
#                         logger.info(f"Extracted JSON: {extracted_json[:500]}...")
#                         response_json = json.loads(extracted_json)
#
#                         # Check for required fields
#                         required_fields = ["ContentMatch", "KeywordMatch", "TotalScore", "MissingKeywords",
#                                            "Skills Gap Analysis"]
#                         for field in required_fields:
#                             if field not in response_json:
#                                 logger.warning(f"Missing required field in Gemini response: {field}")
#                                 if attempt < max_retries - 1:
#                                     await asyncio.sleep(retry_delay)
#                                     continue
#                                 else:
#                                     raise ValueError(f"Missing required field: {field}")
#
#                         return response_json
#                     except json.JSONDecodeError:
#                         if attempt < max_retries - 1:
#                             logger.warning("JSON decode error in extracted text, retrying...")
#                             await asyncio.sleep(retry_delay)
#                             continue
#                         else:
#                             raise HTTPException(
#                                 status_code=status.HTTP_502_BAD_GATEWAY,
#                                 detail="Could not parse JSON content from Gemini response"
#                             )
#                 else:
#                     if attempt < max_retries - 1:
#                         logger.warning("Could not extract JSON pattern, retrying...")
#                         await asyncio.sleep(retry_delay)
#                         continue
#                     else:
#                         raise HTTPException(
#                             status_code=status.HTTP_502_BAD_GATEWAY,
#                             detail="Could not extract valid JSON from Gemini response"
#                         )
#
#             except Exception as e:
#                 logger.error(f"Error processing Gemini response: {str(e)}")
#                 if attempt < max_retries - 1:
#                     await asyncio.sleep(retry_delay)
#                     continue
#                 else:
#                     raise HTTPException(
#                         status_code=status.HTTP_502_BAD_GATEWAY,
#                         detail=f"Error processing Gemini response after retries: {str(e)}"
#                     )
#
#         except Exception as e:
#             logger.error(f"Error with Gemini API: {str(e)}")
#             if attempt < max_retries - 1:
#                 await asyncio.sleep(retry_delay)
#                 continue
#             else:
#                 raise HTTPException(
#                     status_code=status.HTTP_502_BAD_GATEWAY,
#                     detail=f"Gemini API error after retries: {str(e)}"
#                 )
#
#
# # Extract score from various formats
# def extract_score_value(match_percentage):
#     logger.info(f"Extracting score value from: {match_percentage}")
#
#     # Handle if it's already a number
#     if isinstance(match_percentage, (int, float)):
#         return float(match_percentage)
#
#     # Handle if it's a string containing just a number
#     if isinstance(match_percentage, str):
#         try:
#             # Try direct conversion first
#             return float(match_percentage.strip())
#         except ValueError:
#             # If that fails, try to extract a number with regex
#             pass
#
#     # Use regex to find a number pattern in various formats
#     patterns = [
#         r'(\d+\.?\d*)',  # Match numbers like 75 or 75.5
#         r'(\d+\.?\d*)%',  # Match percentage format like 75% or 75.5%
#         r'(\d+\.?\d*)\s*percent',  # Match "75 percent" or "75.5 percent"
#         r'(\d+\.?\d*)\s*out of\s*100',  # Match "75 out of 100"
#         r'(\d+\.?\d*)\s*out of\s*50',  # Match "35 out of 50"
#     ]
#
#     for pattern in patterns:
#         match = re.search(pattern, str(match_percentage))
#         if match:
#             try:
#                 extracted_value = float(match.group(1))
#                 logger.info(f"Successfully extracted score value: {extracted_value}")
#                 return extracted_value
#             except ValueError:
#                 continue
#
#     # If all extraction methods fail, log and return 0
#     logger.warning(f"Could not extract score value from: {match_percentage}, defaulting to 0")
#     return 0.0
#
#
# # Extract scores from response
# def extract_scores(response):
#     logger.info(f"Extracting scores from response")
#
#     # Get the component scores
#     content_score = extract_score_value(response.get("ContentMatch", "0"))
#     keyword_score = extract_score_value(response.get("KeywordMatch", "0"))
#
#     # Get the total score directly or calculate it
#     total_score = extract_score_value(response.get("TotalScore", "0"))
#     calculated_total = content_score + keyword_score
#
#     # If the extracted total seems wrong, use the calculated total
#     if abs(total_score - calculated_total) > 5:  # Allow small differences due to rounding
#         logger.warning(
#             f"Total score ({total_score}) doesn't match sum of components ({calculated_total}), using calculated total")
#         total_score = calculated_total
#
#     # Ensure scores are in proper ranges
#     content_score = max(0, min(50, content_score))
#     keyword_score = max(0, min(50, keyword_score))
#     total_score = max(0, min(100, total_score))
#
#     return {
#         "content_score": content_score,
#         "keyword_score": keyword_score,
#         "total_score": total_score
#     }
#
#
# # Process resume from URL directly with Gemini Vision
# async def process_resume_url(resume_input: ResumeInput, job_description: str, api_key: str):
#     try:
#         url = str(resume_input.url)
#         user_id = resume_input.user_id
#
#         logger.info(f"Processing resume for user: {user_id} from URL: {url}")
#
#         # Generate prompt for image analysis
#         prompt = prepare_image_prompt(url, job_description)
#
#         # Send image URL directly to Gemini Vision API
#         response = await get_gemini_image_response(prompt, url, api_key)
#         logger.info(f"Received analysis for user: {user_id}")
#
#         # Extract scores
#         scores = extract_scores(response)
#
#         # Extract missing keywords
#         missing_keywords = response.get("MissingKeywords", [])
#         if isinstance(missing_keywords, str):
#             # Sometimes the API might return a string instead of a list
#             try:
#                 missing_keywords = json.loads(missing_keywords)
#             except json.JSONDecodeError:
#                 # If string can't be parsed as JSON, convert to list by splitting
#                 missing_keywords = [kw.strip() for kw in missing_keywords.split(',')]
#
#         # Create result with user_id
#         result = {
#             "user_id": user_id,
#             "score": scores["total_score"],
#             "content_score": scores["content_score"],
#             "keyword_score": scores["keyword_score"],
#             "missing_keywords": missing_keywords,
#             "sgAnalysis": response.get("Skills Gap Analysis", "Skills gap analysis not available")
#         }
#
#         logger.info(f"Analysis complete for user: {user_id}, total score: {scores['total_score']}")
#         return result
#
#     except HTTPException as e:
#         # Re-raise HTTP exceptions as is
#         logger.error(f"HTTP exception while processing resume for user {user_id}: {str(e)}")
#         raise e
#     except Exception as e:
#         # Wrap other exceptions
#         logger.error(f"Error processing resume for user {user_id}: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error processing resume for user {user_id}: {str(e)}"
#         )
#
#
# # Process multiple resumes in parallel with rate limiting
# async def process_resumes(resumes: List[ResumeInput], job_description: str, api_key: str):
#     # Use a semaphore to limit concurrent requests
#     semaphore = asyncio.Semaphore(3)  # Process up to 3 resumes concurrently
#
#     async def process_with_semaphore(resume):
#         async with semaphore:
#             try:
#                 return await process_resume_url(resume, job_description, api_key)
#             except Exception as e:
#                 logger.error(f"Error in process_with_semaphore for user {resume.user_id}: {str(e)}")
#                 return {
#                     "user_id": resume.user_id,
#                     "score": 0.0,
#                     "content_score": 0.0,
#                     "keyword_score": 0.0,
#                     "missing_keywords": [],
#                     "sgAnalysis": "Could not analyze skills gap due to processing error."
#                 }
#
#     # Create tasks for all resumes
#     tasks = [process_with_semaphore(resume) for resume in resumes]
#
#     # Execute all tasks
#     results = await asyncio.gather(*tasks)
#
#     # Sort results by score
#     ranked_results = sorted(results, key=lambda x: x["score"], reverse=True)
#
#     return ranked_results
#
#
# # Endpoints
# @app.post("/analyze", response_model=ATSResponse,
#           summary="Analyze resumes against a job description",
#           description="Provide resume image URLs with user IDs and a job description to get ATS analysis")
# async def analyze_resumes_endpoint(
#         request: ATSRequest,
#         api_key: str = Depends(get_api_key)
# ):
#     if not request.resumes:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="No resume URLs provided"
#         )
#
#     logger.info(f"Received analyze request with {len(request.resumes)} resumes")
#     results = await process_resumes(request.resumes, request.job_description, api_key)
#     logger.info(f"Analysis complete for {len(results)} resumes")
#     return {"results": results}
#
#
# # Form-based endpoint for backward compatibility
# @app.post("/analyze-form", response_model=ATSResponse,
#           summary="Analyze resumes against a job description (form-based)",
#           description="Provide resume URLs with user IDs and a job description using form data")
# async def analyze_resumes_form(
#         job_description: str = Form(...),
#         resume_urls: str = Form(...),  # Comma-separated URLs
#         user_ids: str = Form(...),  # Comma-separated user IDs
#         api_key: str = Depends(get_api_key)
# ):
#     # Parse resume URLs and user IDs from form data
#     urls = [url.strip() for url in resume_urls.split(",") if url.strip()]
#     ids = [uid.strip() for uid in user_ids.split(",") if uid.strip()]
#
#     if not urls:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="No resume URLs provided"
#         )
#
#     if len(urls) != len(ids):
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="The number of resume URLs must match the number of user IDs"
#         )
#
#     logger.info(f"Received form-based analyze request with {len(urls)} resumes")
#
#     # Convert to ResumeInput objects with user IDs
#     resume_inputs = [ResumeInput(url=url, user_id=uid) for url, uid in zip(urls, ids)]
#
#     results = await process_resumes(resume_inputs, job_description, api_key)
#     logger.info(f"Form-based analysis complete for {len(results)} resumes")
#     return {"results": results}
#
#
# # Health check endpoint
# @app.get("/health",
#          summary="Health check endpoint",
#          description="Check if the API is running")
# def health_check():
#     return {"status": "healthy"}
#
#
# # Main driver
# if __name__ == "__main__":
#     import uvicorn
#
#     logger.info("Starting ATS Resume Analyzer API server")
#     uvicorn.run(app, host="0.0.0.0", port=8000)