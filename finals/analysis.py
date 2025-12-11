import json
from pathlib import Path
from statistics import median, mean

# Directory containing class*.json files
baseDir = Path(__file__).resolve().parent
dataDir = baseDir / "class_json"

# Load all class JSON files
classes = []

for path in sorted(dataDir.glob("class*.json")):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    classes.append({
        "class_number": data.get("class_number_from_filename"),
        "class_date": data.get("class_date"),
        "attendance": data.get("num_students"),
        "is_exam_day": data.get("is_exam_day", False),
        "exam_type": data.get("exam_type")
    })
# sort so it prints in correct order, glob collects and sorts differently
classes.sort(key=lambda c: c["class_number"])

# 1. number of classes & dates, sorted by class # from above
print("\n1. NUMBER OF CLASSES AND DATES")
print("-----------------------------")
print(f"Total number of classes: {len(classes)}\n")

for c in classes:
    print(f"Class {c['class_number']}: {c['class_date']}")


# 2. median class attendance
attendances = [c["attendance"] for c in classes]

medianAttendance = median(attendances)

print("\n2. MEDIAN CLASS ATTENDANCE")
print("--------------------------")
print(f"Median attendance: {medianAttendance}")

# 3. lowest and highest attendance
minClass = min(classes, key=lambda c: c["attendance"])
maxClass = max(classes, key=lambda c: c["attendance"])

print("\n3. LOWEST & HIGHEST ATTENDANCE")
print("-------------------------------")
print(
    f"Lowest attendance: {minClass['attendance']} "
    f"on {minClass['class_date']} (Class {minClass['class_number']})"
)
print(
    f"Highest attendance: {maxClass['attendance']} "
    f"on {maxClass['class_date']} (Class {maxClass['class_number']})"
)

# 4. exam vs non-exam attendance
examAttendance = [
    c["attendance"]
    for c in classes
    if c["is_exam_day"]
]

nonExamAttendance = [
    c["attendance"]
    for c in classes
    if not c["is_exam_day"]
]

print("\n4. ATTENDANCE VS EXAM DATES")
print("---------------------------")

if examAttendance:
    print(f"Exam dates: {len(examAttendance)}")
    print(f"Average exam-day attendance: {mean(examAttendance):.2f}")
else:
    print("No exam dates detected.")

print(f"Non-exam dates: {len(nonExamAttendance)}")
print(f"Average non-exam attendance: {mean(nonExamAttendance):.2f}")

difference = mean(examAttendance) - mean(nonExamAttendance)
print(f"Difference (exam - non-exam): {difference:.2f}")

# detailed breakdown
print("\nExam day breakdown:")
for c in classes:
    if c["is_exam_day"]:
        print(
            f"- {c['exam_type']} ({c['class_date']}): "
            f"{c['attendance']} students"
        )

# 5. when is attendance highest?
print("\n5. WHEN IS ATTENDANCE HIGHEST?")
print("------------------------------")
print(
    f"Highest attendance occurred on {maxClass['class_date']} "
    f"(Class {maxClass['class_number']}), "
    f"with {maxClass['attendance']} students."
)

if maxClass["is_exam_day"]:
    print(
        f"This was an examination day: {maxClass['exam_type']}."
    )
else:
    print(
        "This was not an examination day."
    )
