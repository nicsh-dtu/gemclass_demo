import os
import time
import json
import requests
import fitz  # PyMuPDF
import re
import pandas as pd

# 1. Extract text from local PDF
def extract_text_from_pdf(file_path):
    """Extract text from a local PDF file."""
    try:
        pdf_document = fitz.open(file_path)
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error processing PDF from {file_path}: {e}")
        return None

# 2. Preprocess text
def preprocess_text(text):
    """Preprocess the text (e.g., normalize and clean)."""
    text = text.lower()
    text = ''.join([c if c.isalnum() or c.isspace() else ' ' for c in text])
    return text.strip()

# 3. Convert LLM output set to dict
def convert_set_to_dict(data):
    """Converts a set containing a single JSON-like string into a dictionary."""
    if isinstance(data, set) and len(data) == 1:
        json_like_string = next(iter(data)).strip("```").strip("json").strip()
        try:
            parsed_dict = json.loads(json_like_string)
            return parsed_dict
        except json.JSONDecodeError:
            return {"error": json_like_string}
    else:
        print(f"Unexpected data type or multiple items in set: {data}")
        return None

# 4. Classify using LLM
def classify_with_llm(paper_text, class_descriptions, gemini_api_key):
    classes_string = "\n".join([f"{cls}: {desc}" for cls, desc in class_descriptions.items()])
    prompt = (
        f"You are an expert in scientific research and classification. Please classify the following research paper into the most relevant classes from the list provided.\n\n"
        f"The classes provided represent knowledge gaps in the scientific field. Your task is to assign these classes to the paper only if the scientific work in the paper directly addresses and contributes to filling these knowledge gaps. Do not fabricate responses or make assumptions beyond what is explicitly or strongly implied in the paper. Assign classes and justifications strictly based on the definitions provided for each class. Please cite specific sections of the paper that support your classification.\n\n"
        f"Classes:\n{classes_string}\n\n"
        f"Research Paper:\n{paper_text}\n\n"
        f"Respond with a dictionary where each relevant class is a key, and the value is the justification for its selection. Do not include any extra text.\n\n"
    )

    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

    try:
        response = requests.post(
            f"{api_url}?key={gemini_api_key}",
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}]}
        )
        response.raise_for_status()
        result = response.json()

        if "candidates" in result and len(result["candidates"]) > 0:
            content = result["candidates"][0].get("content", {}).get("parts", [{}])[0].get("text", "")
            content = content.strip("```").strip("json").strip()
            content = re.sub(r",\s*}", "}", content)
            content = re.sub(r",\s*]", "]", content)
            return {content}
        else:
            return {"error": "Unexpected response structure", "raw_response": result}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}

# 5. Main processor for local files
def get_projectResults(class_descriptions, gemini_api_key, local_folder="Files", wait=0):
    results = {}
    project_results = {}

    for project_name in os.listdir(local_folder):
        project_path = os.path.join(local_folder, project_name)
        if not os.path.isdir(project_path):
            continue

        for idx, pdf_file in enumerate(os.listdir(project_path), start=1):
            if not pdf_file.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(project_path, pdf_file)
            print(f"\nFile {idx}: Processing {pdf_file} in project '{project_name}'...")

            if wait > 0:
                time.sleep(wait)

            text = extract_text_from_pdf(pdf_path)
            if text:
                processed_text = preprocess_text(text)
                if processed_text:
                    classified_classes = classify_with_llm(processed_text, class_descriptions, gemini_api_key)

                    if classified_classes:
                        if isinstance(classified_classes, dict) and "error" in classified_classes:
                            results[pdf_file] = classified_classes
                        else:
                            results[pdf_file] = convert_set_to_dict(classified_classes)
                    else:
                        results[pdf_file] = "Classification failed or invalid response"

                        # Retry logic
                        retry_count = 0
                        while (results[pdf_file] == 0 or len(results[pdf_file]) > 10 or results[pdf_file] == "Classification failed or invalid response") and retry_count < 2:
                            print(f"Retrying classification for {pdf_file} (Attempt {retry_count + 2})...")
                            classified_classes = classify_with_llm(processed_text, class_descriptions, gemini_api_key)
                            if classified_classes:
                                if isinstance(classified_classes, dict) and "error" in classified_classes:
                                    results[pdf_file] = classified_classes
                                else:
                                    results[pdf_file] = convert_set_to_dict(classified_classes)
                            else:
                                results[pdf_file] = "Classification failed or invalid response"
                            retry_count += 1

                        print(f"Total attempts for {pdf_file}: {retry_count + 1}")
                        if results[pdf_file] == "Classification failed or invalid response":
                            results[pdf_file] = {"error": "Classification failed or invalid response after 3 attempts"}
                        elif results[pdf_file] == 0:
                            results[pdf_file] = {"error": "Classification resulted in 0 classes after 3 attempts"}
                        elif len(results[pdf_file]) > 10:
                            results[pdf_file] = {"error": "Classification resulted in more than 10 classes after 3 attempts"}
                    print(f"Total Classes: {len(results[pdf_file])}, Classes: {results[pdf_file]}")
                else:
                    results[pdf_file] = "No text extracted"
            else:
                results[pdf_file] = "No text extracted"

            if project_name not in project_results:
                project_results[project_name] = {}

            project_results[project_name][pdf_file] = results[pdf_file]

    # Remove "error" if other valid classes are present
    for project, pdfs in project_results.items():
        for pdf, classes in pdfs.items():
            if isinstance(classes, dict) and "error" in classes and len(classes) > 1:
                print(f"Removing error class from {pdf}")
                classes.pop("error", None)

    return project_results

def create_classified_df(project_results, class_descriptions, local_folder="Files"):
    data = []

    # Step 1: Build a list of all PDFs and projects
    for project_name in os.listdir(local_folder):
        project_path = os.path.join(local_folder, project_name)
        if not os.path.isdir(project_path):
            continue

        for pdf_file in os.listdir(project_path):
            if not pdf_file.lower().endswith(".pdf"):
                continue

            file_path = os.path.join(project_path, pdf_file)
            data.append({
                'Project Name': project_name,
                'Publication File': file_path,
                'PDF File': pdf_file
            })

    # Step 2: Create initial DataFrame
    df = pd.DataFrame(data)

    # Step 3: Add classification columns
    additional_columns = {
        'Classification not possible': 0,
        'Reason for: Classification not possible': "",
        'Total_Assigned_Knowledge_Gaps': 0,
        'Other_Assigned_Classes': "",
        'Total_Other_Assigned_Classes': 0
    }

    for class_name in class_descriptions.keys():
        additional_columns[class_name] = 0
        additional_columns[f"JUSTIFICATION FOR {class_name}"] = ""

    for col, val in additional_columns.items():
        df[col] = val

    # Step 4: Fill in classification results
    for index, row in df.iterrows():
        project_name = row['Project Name']
        pdf_file = row['PDF File']

        if project_name in project_results and pdf_file in project_results[project_name]:
            classifications = project_results[project_name][pdf_file]
            if classifications is not None:
                if "error" in classifications:
                    df.at[index, 'Classification not possible'] = 1
                    df.at[index, 'Reason for: Classification not possible'] = classifications["error"]
                else:
                    total_assigned = 0
                    other_assigned_classes = {}
                    for class_name in classifications:
                        if class_name in class_descriptions:
                            df.at[index, class_name] = 1
                            df.at[index, f"JUSTIFICATION FOR {class_name}"] = classifications[class_name]
                            total_assigned += 1
                        else:
                            other_assigned_classes[class_name] = classifications[class_name]
                    df.at[index, 'Total_Assigned_Knowledge_Gaps'] = total_assigned
                    df.at[index, 'Other_Assigned_Classes'] = json.dumps(other_assigned_classes)
                    df.at[index, 'Total_Other_Assigned_Classes'] = len(other_assigned_classes)
            else:
                df.at[index, 'Classification not possible'] = 1
                df.at[index, 'Reason for: Classification not possible'] = "Classification failed or invalid response"
        else:
            df.at[index, 'Classification not possible'] = 1
            df.at[index, 'Reason for: Classification not possible'] = "No classification result found"

    return df

def replace_wrong_classes_fromDict(wrong_classes_mapping, project_results):
    for project, pdfs in project_results.items():
        # Iterate over each PDF in the project
        for pdf, classifications in pdfs.items():
            if classifications is not None and "error" not in classifications:
                keys_to_remove = []
                keys_to_add = {}
                for key, value in classifications.items():
                    for correct_class, wrong_classes in wrong_classes_mapping.items():
                        if key in wrong_classes:
                            print(f"{key} -> {correct_class}")
                            # Add the wrong class key to the list of keys to remove
                            keys_to_remove.append(key)
                            # Prepare the correct class with the same value
                            if correct_class in classifications:
                                keys_to_add[correct_class] = classifications[correct_class] + f"; [Additionally (wrongly) classified as {key} ] {value}"
                            else:
                                keys_to_add[correct_class] = value
                            # ALTERNATIVELY (to not story so many descriptions), we could use the first description only
                            # by replacing the if else statement from above with:: if correct_class not in classification: keys_to_add[correct_class] = value

                # Remove the wrong class keys
                for key in keys_to_remove:
                    del classifications[key]

                # Add the correct class keys
                for key, value in keys_to_add.items():
                    classifications[key] = value
    total_removed_classes = 0


def remove_phrases_from_dict(strings_to_check, removed_project_results, project_results):
    for project, pdfs in project_results.items():
        for pdf, classes in pdfs.items():
            if isinstance(classes, dict) and "error" not in classes:
                keys_to_remove = []
                for cls, justification in classes.items():
                    if justification is None or any(s.lower() in justification.lower() for s in strings_to_check):
                        print(f"Project: {project}, PDF: {pdf}, Class: {cls}, Justification: {justification}")
                        if project not in removed_project_results:
                            removed_project_results[project] = {}
                        if pdf not in removed_project_results[project]:
                            removed_project_results[project][pdf] = {}
                        removed_project_results[project][pdf][cls] = justification
                        keys_to_remove.append(cls)
                for key in keys_to_remove:
                    del classes[key]
                # Update the justification in the project_results
                for cls, justification in classes.items():
                    if isinstance(justification, dict) and "justification" in justification:
                        classes[cls] = justification["justification"]