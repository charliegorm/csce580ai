import os
import re
import json
import base64 
from pathlib import Path

from openai import OpenAI
# uses OPENAI_API_KEY from my environment (set in terminal)
client = OpenAI()

# all relative to this script's location
baseDir = Path(__file__).resolve().parent
imageDir = baseDir / "class_data"
outDir = baseDir / "class_json"

# exam dates in ISO format
examDates = {
    "2025-10-07": "Quiz 2",
    "2025-11-11": "Quiz 3",
    "2025-11-18": "Presentation",
}

# using a stronger vision model for better counting accuracy, gpt-4o is vision-capable
modelName = "gpt-4o" 


def encodeImageToBase64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def annotateExamInfo(record: dict) -> dict:
    """Add is_exam_day / exam_type based on class_date."""
    date = record.get("class_date")
    examType = examDates.get(date)
    record["is_exam_day"] = examType is not None
    record["exam_type"] = examType
    return record


def extractAttendanceFromImage(image_path: Path) -> dict:
    """
    Call the LLM to extract anonymized attendance info from a single image.
    """
    imgB64 = encodeImageToBase64(image_path)

    prompt = """
You are analyzing a clear photo of a handwritten CSCE580 class attendance sheet.

Your job is to return a JSON object with anonymized attendance data.
IMPORTANT:
- Do NOT include any real student names or usernames in the JSON.
- The MOST IMPORTANT thing is to accurately count **all** student rows.

GENERAL SHAPE OF THE SHEET:
- There is usually a header at the top (e.g., "Class 1 - 19 Aug 2025").
- Below that, there are one or two vertical columns of rows where students write their name and username.
- There may be a header row like "Name / Username" above the student rows.
- Some sheets have two side-by-side columns of students; you MUST count both columns.

INSTRUCTIONS (FOLLOW CAREFULLY):

1. HEADER PARSING
   - Read the header text at the top (e.g. "Class 1 - 19 Aug 2025").
   - Store the full header text in "class_title_raw".
   - Extract the class number if present into "class_number".
   - Extract the class date in ISO format YYYY-MM-DD into "class_date".
     * If the month is given as a name (e.g., "Aug", "October"), convert it.
     * If you are unsure, set "class_date" to null.

2. IDENTIFYING STUDENT ROWS (KEY PART)
   - A **student row** is a horizontal line where a student has written something in the "Name" and/or "Username" area.
   - DO NOT count header rows such as "Name / Username" or any obvious non-student notes.
   - INCLUDE rows even if:
       * The handwriting is messy or partially illegible.
       * Only part of the name or username is visible.
       * There are checkmarks or short scribbles instead of a full name.
       * The row is partially cut off at the edge of the image but clearly corresponds to a student line.
   - If the sheet has TWO COLUMNS of students, scan **both columns from top to bottom** and include every row in both columns that looks like a student entry.
   - If you are uncertain whether a row is a student or not, but it appears to be in the main list of names/usernames with some writing on it, **treat it as a student row**.

3. ANONYMIZATION
   - For each student row you detect:
       - Assign a label "student_1", "student_2", ..., in strict top-to-bottom, left-to-right order.
       - For each student, output:
           "student_label": "student_N"
           "username_label": "student_N_user"
   - DO NOT output their real name or username anywhere.
   - You do NOT need to reconstruct the actual username string; just use the anonymized labels.

4. CONSISTENT COUNTING
   - Let the number of student rows you detect be N.
   - The "students" array must contain **exactly N** elements, one for each student.
   - The field "num_students" must be exactly equal to N.
   - BEFORE returning the JSON, double-check:
       - You did not accidentally include header rows or blank lines.
       - You did not skip any row that appears to be a student entry in either column.
       - "num_students" == length of "students" array.

5. JSON FORMAT
Return ONLY a JSON object with this exact structure (no extra keys):

{
  "class_title_raw": string,
  "class_number": integer or null,
  "class_date": string in "YYYY-MM-DD" format or null,
  "num_students": integer,
  "students": [
    {
      "student_label": "student_1",
      "username_label": "student_1_user"
    },
    ...
  ]
}
"""

    # calling Chat Completions API with JSON response_format, with my key there are sufficient funds supplied
    completion = client.chat.completions.create(
        model=modelName,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{imgB64}"
                        },
                    },
                ],
            }
        ],
    )

    content = completion.choices[0].message.content
    data = json.loads(content)

    # ensuring num_students matches length of students
    students = data.get("students", [])
    data["num_students"] = len(students)

    return data


def main():
    outDir.mkdir(parents=True, exist_ok=True)

    # looping over all class*.jpg images
    for imgPath in sorted(imageDir.glob("class*.jpg")):
        print(f"Processing {imgPath.name} ...")

        # using filename to get class_number_from_filename
        m = re.search(r"class(\d+)", imgPath.stem, re.IGNORECASE)
        classNumFromFile = int(m.group(1)) if m else None

        # calling LLM to extract data
        record = extractAttendanceFromImage(imgPath)

        # attaching extra metadata
        record["image_filename"] = imgPath.name
        record["class_number_from_filename"] = classNumFromFile

        # Annotate exam-day info based on class_date
        record = annotateExamInfo(record)

        # Decide output filename (use the number from the filename)
        if classNumFromFile is not None:
            outName = f"class{classNumFromFile}.json"
        else:
            # Fallback to using the image stem
            outName = imgPath.with_suffix(".json").name

        outPath = outDir / outName
        with open(outPath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        print(f"\t-> wrote {outPath}")

    print("Done.")


if __name__ == "__main__":
    main()
