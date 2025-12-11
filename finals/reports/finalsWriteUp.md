# Charlie Gorman CSCE580 Finals Write-Up
## Q1a. Graduate Paper Comprehension

**Q1a.a - Write the name of the paper and student presenter you chose.**
**Title:** Understanding Emotional Body Expressions via Large Language Models 
**Presenter:** Yamuna Bobbala. 
**Q1a.b - Now, can you think and create a new example exemplifying the main conclusion of the paper.**
**Example Input Scenario:** A person is seen crouched against a wall, with their back turned to the viewer and head leaning against the wall. Fists are clenched at this person's sides and suddenly this person strikes the wall with their fist. 
**Example Output Scenario:** Emotion: Anger. Explanation: Although the person's face is not seen, their fists are clenched, their posture is defeated, and the striking of the wall indicates frustration or agitation, which are both consistent with anger. 
**Q1a.c - Describe how the conclusion is supported in your example.**
This example supports the paper's conclusion because an AI model is making emotional diagnoses or classifications not based on tone of voice or facial expressions, but based upon the skeletal motion / body language of individuals. The movements and positions of the person are tokenized and processed into factors that contribute to the emotional state of an individual. 

## Q2. Using AI for tackling a pressing teaching problem – classroom absenteeism. 

**Q2.a - Describe your data preparation, if any, and why or why not.**
My data preparation included downloading all of the .jpg files of the class attendance sheets. I sorted through the files manually, compared them against each other, and removed the few duplicate files. These duplicate files had different file names but had nearly identical, if not exactly identical, content within. I manually renamed the files based on which class they were from, as a way to make things more uniform. I did this step mostly for my own clarity, to ensure that there were no more duplicates and that lecture 23 was missing. It also made traversing the files in order to convert them to .json files much easier. I introduced anonymity to the data, replacing student names and usernames with vanilla fillers (student_1, student_1_user, etc.), for anonymity sake but also because the individuals were not the focus of this. Finally, once all of these .png's were converted into separate .json files, I went through them all to make sure the LLM (gpt-4o) didn't make mistakes in counting the students attended. In the few cases where the LLM failed, I corrected the number of students present. 
**Q2.b - Describe your steps to create a model – pre-trained, your own, manual.**
I used a pre-trained visual-capable LLM (gpt-4o). I decided to use this model because I didn't want to deal with spending the time on training a model, to submit this assignment on time. Also, because out of the OpenAI APIs, this one performs best with reading handwritten data, performs quickest, and with the least cost. 
**Q2.c - Answer the questions from your analyses using the models:**
- Q2.c.a What are the number of classes and their dates?
Shown in my sample output of my analysis.py script, below. The classes are ordered and present their respective dates in the same format throughout.
- Q2.c.b What is the median class attendance per class?
33.0 students
- Q2.c.c What are the dates with lowest and highest attendance?

3. LOWEST & HIGHEST ATTENDANCE
-------------------------------
Lowest attendance: 16 on 2025-11-20 (Class 27)
Highest attendance: 49 on 2025-08-21 (Class 2)

- Q2.c.d Is there a correlation of high attendance with course evaluations dates?
Although the day of most students attending was not on an exam-day, I believe there is a correlation of high attendance with course evaluation dates. As shown in my sample output below:
4. ATTENDANCE VS EXAM DATES
---------------------------
Exam dates: 2
Average exam-day attendance: 43.00
Non-exam dates: 24
Average non-exam attendance: 32.83
Difference (exam - non-exam): 10.17.
I believe that this consistent increase of student attendance (+10.17 students) across all exam days indicates a positive correlation between exam dates and student attendance. 

**Q2.d - If you had more time(say a week), what more could you have done to improve performance?**
If I had more time, I would have given Fine-Tuned DistilBERT another shot. Last time (and my first time) implementing this model for ProjectB, I used only one epoch to save time and to save my machine from immense computational work. This time around I would have tried to figure out how to run the training, testing, and validation of a Fine-Tuned DistilBERT model in a more efficient manner for my machine. I believe performance would have done better due to training for this specific task, as opposed to using a generic model which is widely-available. 

## **Sample output of analysis.py:**

/Users/charliegorman/Desktop/csce/csce580/.venv/bin/python /Users/charliegorman/Desktop/csce/csce580/finals/analysis.py

1. NUMBER OF CLASSES AND DATES
-----------------------------
Total number of classes: 26

Class 1: 2025-08-19
Class 2: 2025-08-21
Class 3: 2025-08-26
Class 4: 2025-08-28
Class 5: 2025-09-02
Class 6: 2025-09-04
Class 7: 2023-09-09
Class 8: 2025-09-11
Class 9: 2025-09-16
Class 10: 2025-09-18
Class 11: 2025-09-28
Class 12: 2025-09-25
Class 13: 2025-09-30
Class 14: 2025-10-02
Class 15: 2025-10-07
Class 16: 2025-10-14
Class 17: 2025-10-16
Class 18: 2025-10-21
Class 19: 2025-10-23
Class 20: 2021-10-28
Class 21: 2025-10-30
Class 22: 2025-11-04
Class 24: 2025-11-11
Class 25: 2025-11-13
Class 26: 2023-11-18
Class 27: 2025-11-20

2. MEDIAN CLASS ATTENDANCE
--------------------------
Median attendance: 33.0

3. LOWEST & HIGHEST ATTENDANCE
-------------------------------
Lowest attendance: 16 on 2025-11-20 (Class 27)
Highest attendance: 49 on 2025-08-21 (Class 2)

4. ATTENDANCE VS EXAM DATES
---------------------------
Exam dates: 2
Average exam-day attendance: 43.00
Non-exam dates: 24
Average non-exam attendance: 32.83
Difference (exam - non-exam): 10.17

Exam day breakdown:
- Quiz 2 (2025-10-07): 45 students
- Quiz 3 (2025-11-11): 41 students

5. WHEN IS ATTENDANCE HIGHEST?
------------------------------
Highest attendance occurred on 2025-08-21 (Class 2), with 49 students.
This was not an examination day.