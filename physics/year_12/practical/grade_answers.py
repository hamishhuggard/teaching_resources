#!/usr/bin/env python3
import argparse
import os
import sys
import re
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types, errors

# =====================================================================
# HARDCODED CONFIGURATION
# =====================================================================

# Reference files are located in the "assessment_files" subdirectory.
# Replace the .doc file with a .pdf or .txt version before running!
REFERENCE_FILES = [
    "assessment_files/AS 91168 Clarification.txt",
    "assessment_files/physics2_1B_v3_NZQA 151119.pdf", # Converted from .doc
    "assessment_files/as91168.pdf",
    "assessment_files/lvl 2 practical investigation.pdf",
    "assessment_files/91168-EXP-student1-001.pdf",
    # "assessment_files/91168-EXP-student2-001.pdf",
    # "assessment_files/91168-EXP-student3-001.pdf",
    # "assessment_files/91168-EXP-student4-001.pdf",
    # "assessment_files/91168-EXP-student5-001.pdf"
]

PREAMBLE = """
Role: You are "Physics Practical Investigation Helper (Year 12)"

General Instructions: Your job is to grade the student's level 2 NCEA physics practical investigation based on the reference materials provided. 

Context of the experiment:
The instructions for each experiment are given to them in exam.net. They should repeat each measurement 2-3 times, for 3-4 trials total. 4 is ideal but 3 is ok if there's not enough time. We are using Google Sheets for tabulating and plotting data. The table should have columns something like: independent variable, dependent variable trial 1, trial 2, trial 3, trial 4, average, then the x-axis variable raised to a power (either -2, -1, 0.5, or 2). There may be additional columns for intermediate calculations. Note that the x-axis will not necessarily be the independent variable. We choose the x-axis so that the gradient physically represents something like a spring constant or gravity. The student then needs to create a chart with a sensible title and axis labels (with correct units) and get exponent of the relationship using google sheets "trend line" in "power series" mode. The power of this trend line should be rounded to the nearest option of 2, 0.5, -1, -2. This will be the canonical power we will use. The student should copy paste the parameters of this trend from the plot into their report. We then raise the x-axis variable in the last column of the table to this power and plot y against x^n with a linear trend line. The student should copy paste the parameters of this trend from the plot into their report.

Grading Instructions:
1. Make sure students know the correct units at every step. 
2. Help them address as many points as they can to get a high grade, but prioritize the achievement criteria first. Prompt them to think about why they choose a maximum and minimum value for the independent variable etc.
3. They should link the experiment to existing physics. They have already completed the ncea level 2 mechanics topic, so should know this stuff. 
4. Assess their answers based on the rubric.
5. For each answer, inject inline annotations by adding exactly this format: [ANNOTATION: your comment here]. Do not alter their original text, just insert these annotations where appropriate to point out errors or good points. 
5b. If some particular sentence would earn an E or M or A point, then say add an annotation like "A. [explanation]"
6. Provide an overall feedback summary and a grade of N (Not Achieved), A (Achieved), M (Merit), or E (Excellence).
7. Focus on giving strategy advice about how the student can improve for the next experiment. Minor errors need minimal commentary.
8. In the per-question feedback, explain how the student could have gotten to a perfect answer. Show them an improved version of their answer which would get maximum points. 
9. For the overall feedback, give itemized bulletpoints for what the student should change next time to improve their grade (in descending order of impact)
10. There are three questions about control variables, but 1 is fine. Don't tell students they need to describe more control variables.
"""

# =====================================================================
# SCHEMA DEFINITIONS FOR API
# =====================================================================
class QuestionAssessment(BaseModel):
    number: str = Field(description="The question number, e.g., q1, q2")
    annotated_answer: str = Field(description="The exact original student answer, but with [ANNOTATION: comment] tags injected inline.")
    feedback: str = Field(description="General feedback for this specific question.")
    grade: str = Field(description="Must be exactly one of: N, A, M, E")

class GradingResult(BaseModel):
    question_assessments: List[QuestionAssessment]
    overall_feedback: str
    overall_grade: str = Field(description="Must be exactly one of: N, A, M, E")


def find_assessment(result, q_num, position):
    """
    Match a QuestionAssessment to a question row by exact q_num or by positional index.
    """
    q_num_norm = str(q_num).strip().lower()
    for item in result.question_assessments:
        if str(item.number).strip().lower() == q_num_norm:
            return item
    if 0 <= position < len(result.question_assessments):
        return result.question_assessments[position]
    return None


# =====================================================================
# MAIN LOGIC
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Grade student answers using Gemini API.")
    parser.add_argument("-i", "--input", required=True, help="Input CSV file")
    parser.add_argument("-o", "--output", help="Output directory. Defaults to input filename sans .csv")
    parser.add_argument("-l", "--limit", type=int, help="Stop after grading this many NEW students.")
    args = parser.parse_args()

    # Determine output directory
    input_csv = args.input
    out_dir = args.output if args.output else os.path.splitext(os.path.basename(input_csv))[0]

    os.makedirs(out_dir, exist_ok=True)

    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not found. Please add it to your .env file.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client()
    
    # Upload reference PDFs
    uploaded_files = []
    print("Uploading reference files to Gemini API...")
    for file_path in REFERENCE_FILES:
        if os.path.exists(file_path):
            print(f"  Uploading {file_path}...")
            uploaded_file = client.files.upload(file=file_path)
            uploaded_files.append(uploaded_file)
        else:
            print(f"  [WARNING] File not found and will be skipped: {file_path}", file=sys.stderr)

    # Read CSV
    df = pd.read_csv(input_csv)
    
    # DYNAMIC COLUMN DETECTION (Fixes the issue where Student cols were treated as text)
    first_student_idx = -1
    for i, col in enumerate(df.columns):
        if 'student' in col.lower():
            first_student_idx = i
            break
            
    # If we couldn't find a column with "student" in the name, fallback to assuming 
    # the first column is metadata and the rest are students.
    if first_student_idx == -1 or first_student_idx == 0:
        first_student_idx = 1
        
    std_cols = df.columns[:first_student_idx].tolist()
    student_cols = df.columns[first_student_idx:].tolist()
    
    print(f"Detected Metadata Columns: {std_cols}")
    print(f"Detected Student Columns: {len(student_cols)} students found.")
    
    student_results = {}
    skip_students = set()
    prev_df = None
    out_csv_path = os.path.join(out_dir, "graded_answers.csv")

    # Resume state logic
    if os.path.exists(out_csv_path):
        print(f"Found existing output at {out_csv_path}. Attempting to resume...")
        prev_df = pd.read_csv(out_csv_path)
        # Find the overall grade row to determine who is already fully graded
        overall_g_rows = prev_df[prev_df[std_cols[0]].astype(str) == 'overall_grade']
        if not overall_g_rows.empty:
            overall_g_row = overall_g_rows.iloc[-1]
            for student in student_cols:
                if student in prev_df.columns:
                    val = str(overall_g_row.get(student, "")).strip()
                    if val in ['N', 'A', 'M', 'E']:
                        skip_students.add(student)
        print(f"Resuming: {len(skip_students)} students already graded. {len(student_cols) - len(skip_students)} remaining.")

    print("\nStarting grading process...")
    graded_count = 0
    for idx, student in enumerate(student_cols):
        if student in skip_students:
            print(f"Skipping student {idx+1}/{len(student_cols)} (Already graded)...")
            continue

        if args.limit and graded_count >= args.limit:
            print(f"\nLimit of {args.limit} new students reached. Stopping.")
            break

        print(f"Grading student {idx+1}/{len(student_cols)} (Anonymized for API)...")
        
        # Build prompt
        prompt_text = f"{PREAMBLE}\n\nHere are the answers for a single student. Please assess them.\n\n"
        
        for q_idx, row in df.iterrows():
            q_num = f"q{q_idx+1}" # Explicit internal numbering to prevent confusion
            ans = str(row[student]) if pd.notna(row[student]) else "[No answer provided]"
            
            prompt_text += f"--- Question {q_num} ---\n"
            # Append whatever metadata columns we have (Title, Text, etc.)
            for meta_col in std_cols:
                prompt_text += f"{meta_col}: {row[meta_col]}\n"
            prompt_text += f"Student Answer:\n{ans}\n\n"

        prompt_text += (
            "IMPORTANT: In your JSON response, the 'number' field for each question assessment must be "
            "the exact identifier shown right after the word 'Question' above (e.g. 'q1', 'q2', 'q3'). "
            "Do NOT use the numbering or lettering that appears inside the question title itself.\n"
        )

        contents = uploaded_files + [prompt_text]
        
        try:
            # Model easily switchable via comments
            response = client.models.generate_content(
                model='gemini-3.1-flash-lite',
                # model='gemini-2.5-flash',
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GradingResult,
                    temperature=0.2, 
                ),
            )
            
            # Parse output
            result_data = GradingResult.model_validate_json(response.text)
            student_results[student] = result_data
            graded_count += 1
        except Exception as e:
            print(f"\n[ERROR] API request failed while grading student {idx+1}: {e}")
            print("Halting grading gracefully and proceeding to save partial results...")
            break

    # =====================================================================
    # CSV RECONSTRUCTION
    # =====================================================================
    print("\nReconstructing CSV...")
    new_rows = []
    
    for q_idx, (_, row) in enumerate(df.iterrows()):
        q_num = f"q{q_idx+1}"
        new_rows.append(row.to_dict())
        
        # Initialize dynamically based on how many std_cols exist
        f_row = {col: "" for col in std_cols}
        g_row = {col: "" for col in std_cols}
        
        f_row[std_cols[0]] = f"{q_num}_feedback"
        if len(std_cols) > 1: f_row[std_cols[1]] = "Feedback"
        
        g_row[std_cols[0]] = f"{q_num}_grade"
        if len(std_cols) > 1: g_row[std_cols[1]] = "Grade"
        
        for student in student_cols:
            if student in student_results:
                assessment = find_assessment(student_results[student], q_num, q_idx)
                if assessment:
                    f_row[student] = assessment.feedback
                    g_row[student] = assessment.grade
                else:
                    f_row[student] = "ERROR"
                    g_row[student] = "N/A"
            elif student in skip_students and prev_df is not None:
                # Splice in data from the previous run using our explicit tags
                prev_f = prev_df[prev_df[std_cols[0]].astype(str) == f"{q_num}_feedback"]
                prev_g = prev_df[prev_df[std_cols[0]].astype(str) == f"{q_num}_grade"]
                f_row[student] = prev_f.iloc[0][student] if not prev_f.empty else "ERROR"
                g_row[student] = prev_g.iloc[0][student] if not prev_g.empty else "N/A"
            else:
                f_row[student] = "UNGRADED"
                g_row[student] = "N/A"
                
        new_rows.append(f_row)
        new_rows.append(g_row)
        
    overall_f_row = {col: "" for col in std_cols}
    overall_f_row[std_cols[0]] = 'overall_feedback'
    if len(std_cols) > 1: overall_f_row[std_cols[1]] = 'Overall Feedback'
    
    overall_g_row = {col: "" for col in std_cols}
    overall_g_row[std_cols[0]] = 'overall_grade'
    if len(std_cols) > 1: overall_g_row[std_cols[1]] = 'Overall Grade'
    
    for student in student_cols:
        if student in student_results:
            overall_f_row[student] = student_results[student].overall_feedback
            overall_g_row[student] = student_results[student].overall_grade
        elif student in skip_students and prev_df is not None:
            prev_overall_f = prev_df[prev_df[std_cols[0]].astype(str) == 'overall_feedback']
            prev_overall_g = prev_df[prev_df[std_cols[0]].astype(str) == 'overall_grade']
            overall_f_row[student] = prev_overall_f.iloc[-1][student] if not prev_overall_f.empty else "ERROR"
            overall_g_row[student] = prev_overall_g.iloc[-1][student] if not prev_overall_g.empty else "N/A"
        else:
            overall_f_row[student] = "UNGRADED"
            overall_g_row[student] = "N/A"
            
    new_rows.append(overall_f_row)
    new_rows.append(overall_g_row)
    
    out_csv_path = os.path.join(out_dir, "graded_answers.csv")
    out_df = pd.DataFrame(new_rows)
    out_df.to_csv(out_csv_path, index=False)
    print(f"Saved graded CSV to {out_csv_path}")

    # =====================================================================
    # HTML GENERATION
    # =====================================================================
    html_dir = os.path.join(out_dir, "html_reports")
    os.makedirs(html_dir, exist_ok=True)
    
    def grade_color_class(grade):
        return f"grade-{grade.upper()}" if grade.upper() in ['N', 'A', 'M', 'E'] else "grade-N"
        
    for student in student_cols:
        if student in skip_students:
            continue
            
        if student not in student_results:
            continue
            
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Feedback Report</title>
            <style>
                body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #2d3748; line-height: 1.6; }}
                h1 {{ color: #1a202c; border-bottom: 2px solid #e2e8f0; padding-bottom: 1rem; }}
                h3 {{ color: #2c3e50; margin-top: 0; }}
                .question-block {{ border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.5rem; margin-bottom: 2rem; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
                .q-text {{ font-style: italic; color: #4a5568; margin-bottom: 1.5rem; border-left: 4px solid #cbd5e0; padding-left: 1rem; }}
                .answer-block {{ background: #f7fafc; padding: 1.5rem; border-radius: 6px; white-space: pre-wrap; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin-bottom: 1rem; border: 1px solid #edf2f7; color: #1a202c; line-height: 1.8; }}
                .annotation {{ background-color: #fefcbf; color: #975a16; padding: 2px 6px; border-radius: 4px; font-weight: 600; font-size: 0.9em; margin: 0 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); border: 1px solid #f6e05e; display: inline-block; }}
                .feedback-box {{ background: #ebf8ff; padding: 1.25rem; border-radius: 6px; border-left: 4px solid #4299e1; margin-top: 1rem; }}
                .grade-badge {{ display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: bold; font-size: 0.875rem; color: white; margin-left: 0.5rem; }}
                .grade-E {{ background-color: #48bb78; }}
                .grade-M {{ background-color: #ecc94b; color: #744210; }}
                .grade-A {{ background-color: #ed8936; }}
                .grade-N {{ background-color: #f56565; }}
                .overall-block {{ background: #2d3748; color: white; padding: 2rem; border-radius: 8px; margin-top: 3rem; }}
                .overall-block h2 {{ color: white; margin-top: 0; border-bottom: 1px solid #4a5568; padding-bottom: 1rem; }}
                .overall-feedback {{ background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 6px; white-space: pre-wrap; margin-bottom: 1.5rem; line-height: 1.8; }}
                .big-grade {{ font-size: 1.25rem; padding: 0.5rem 1rem; }}
            </style>
        </head>
        <body>
            <h1>Practice Assessment: {student}</h1>
        """
        
        for q_idx, (_, row) in enumerate(df.iterrows()):
            q_num = f"q{q_idx+1}"
            assessment = find_assessment(student_results[student], q_num, q_idx)
            
            # Safely grab title and text based on what columns exist in your CSV
            q_title = str(row[std_cols[0]]) if len(std_cols) > 0 else f"Question {q_idx+1}"
            q_text = str(row[std_cols[1]]) if len(std_cols) > 1 else ""
            
            html_content += f"""
            <div class="question-block">
                <h3>{q_title} ({q_num})</h3>
                <div class="q-text">{q_text}</div>
            """
            
            if assessment:
                annotated_text = assessment.annotated_answer
                annotated_text = annotated_text.replace('<', '&lt;').replace('>', '&gt;')
                annotated_text = re.sub(
                    r'\[ANNOTATION:\s*(.*?)\]', 
                    r'<span class="annotation">📝 \1</span>', 
                    annotated_text,
                    flags=re.IGNORECASE
                )
                
                html_content += f"""
                <div class="answer-block">{annotated_text}</div>
                <div class="feedback-box">
                    <strong>Feedback:</strong> {assessment.feedback}
                    <br><br>
                    <strong>Grade:</strong> <span class="grade-badge {grade_color_class(assessment.grade)}">{assessment.grade}</span>
                </div>
                """
            else:
                html_content += f"""<div class="answer-block">{row[student]}</div>"""
            html_content += "</div>"
            
        res = student_results[student]
        html_content += f"""
            <div class="overall-block">
                <h2>Overall Assessment</h2>
                <div class="overall-feedback">{res.overall_feedback}</div>
                <div>
                    <strong>Final Grade:</strong> 
                    <span class="grade-badge big-grade {grade_color_class(res.overall_grade)}">{res.overall_grade}</span>
                </div>
            </div>
        </body>
        </html>
        """
        
        safe_student_name = "".join([c for c in student if c.isalpha() or c.isdigit() or c==' ']).rstrip()
        with open(os.path.join(html_dir, f"{safe_student_name}_report.html"), "w", encoding="utf-8") as f:
            f.write(html_content)
            
    print(f"Saved generated HTML reports to {html_dir}/")
    
    # Cleanup files
    if uploaded_files:
        print("Cleaning up uploaded files from Gemini API...")
        for f in uploaded_files:
            client.files.delete(name=f.name)

if __name__ == "__main__":
    main()