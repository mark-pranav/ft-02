# test_client.py - Client script to test the FastAPI ATS service

import requests
import time
import json
import os
import sys

# API endpoint (change this if you're running the server on a different host/port)
API_URL = "http://localhost:8000"


def submit_job(resumes_paths, job_description, api_key):
    """Submit a job to the API."""
    url = f"{API_URL}/analyze-resumes/"

    # Prepare files for upload
    files = [('resumes', (os.path.basename(path), open(path, 'rb'), 'application/pdf'))
             for path in resumes_paths]

    # Add form data
    data = {
        'job_description': job_description,
        'api_key': api_key
    }

    # Send request
    response = requests.post(url, files=files, data=data)

    # Check for errors
    if response.status_code != 200:
        print(f"Error submitting job: {response.text}")
        sys.exit(1)

    # Parse response
    job_data = response.json()
    job_id = job_data["job_id"]

    print(f"Job submitted successfully! Job ID: {job_id}")
    return job_id


def check_job_status(job_id):
    """Check the status of a job."""
    url = f"{API_URL}/job-status/{job_id}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error checking job status: {response.text}")
        return None

    return response.json()


def poll_until_complete(job_id, interval=5, timeout=300):
    """Poll the job status until it completes or times out."""
    start_time = time.time()

    while (time.time() - start_time) < timeout:
        job_status = check_job_status(job_id)

        if not job_status:
            print("Failed to get job status")
            return None

        # Print progress
        print(
            f"Status: {job_status['status']} - Processed {job_status['processed_resumes']} of {job_status['total_resumes']} resumes")

        # Check if job is complete
        if job_status['status'] == 'completed':
            print("Job completed successfully!")
            return job_status

        # Check if job failed
        if job_status['status'] == 'failed':
            print(f"Job failed: {job_status['error']}")
            return job_status

        # Wait before polling again
        time.sleep(interval)

    print(f"Timeout after {timeout} seconds")
    return None


def main():
    # Get API key
    api_key = input("Enter your Gemini API key: ")

    # Get job description
    print("\nEnter the job description (type 'END' on a new line when finished):")
    job_description_lines = []
    while True:
        line = input()
        if line.strip().upper() == 'END':
            break
        job_description_lines.append(line)

    job_description = "\n".join(job_description_lines)

    # Get resume paths
    print("\nEnter paths to PDF resumes (one per line, type 'END' when finished):")
    resume_paths = []
    while True:
        path = input()
        if path.strip().upper() == 'END':
            break
        if os.path.exists(path) and path.lower().endswith('.pdf'):
            resume_paths.append(path)
        else:
            print(f"Warning: {path} is not a valid PDF file path")

    if not resume_paths:
        print("No valid resume paths entered. Exiting.")
        return

    # Submit job
    job_id = submit_job(resume_paths, job_description, api_key)

    # Poll for results
    job_results = poll_until_complete(job_id)

    if job_results and job_results['status'] == 'completed':
        # Display results
        print("\n" + "=" * 60)
        print("RESUME RANKINGS (by match score)")
        print("=" * 60)

        # Sort results by score
        sorted_results = sorted(job_results['results'], key=lambda x: x['score'], reverse=True)

        for i, result in enumerate(sorted_results):
            print(f"{i + 1}. {result['resume_name']}: {result['score']}%")

        # Ask if user wants detailed results
        show_details = input("\nShow detailed analysis? (y/n): ").lower() == 'y'

        if show_details:
            print("\n" + "=" * 60)
            print("DETAILED ANALYSIS")
            print("=" * 60)

            for result in sorted_results:
                print(f"\n--- {result['resume_name']} - Score: {result['score']}% ---")
                print("Missing Keywords:")
                print(", ".join(result['missing_keywords']) if result['missing_keywords'] else "None")
                print("\nProfile Summary:")
                print(result['profile_summary'])
                print("-" * 80)

        # Ask if user wants to save results
        save_results = input("\nSave results to file? (y/n): ").lower() == 'y'

        if save_results:
            filename = input("Enter filename (default: ats_results.json): ").strip() or "ats_results.json"
            with open(filename, 'w') as f:
                json.dump(sorted_results, f, indent=2)
            print(f"Results saved to {filename}")


if __name__ == "__main__":
    main()