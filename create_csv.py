import os
import csv

# --- CONFIGURATION ---
image_folder_path = r"D:\Major Project stuff\project_code\package\FINAL_CLEAN_DATASET v2"
output_csv_file = "labels.csv"


# ---------------------

def create_image_csv_by_date(folder_path, csv_path):
    print("Step 1: Creating an ordered list of image filenames...")
    image_extensions = ('.jpg', '.jpeg', '.png')
    try:
        files_with_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
        image_files = [f for f in files_with_paths if f.lower().endswith(image_extensions)]
        image_files.sort(key=os.path.getmtime)
        filenames = [os.path.basename(f) for f in image_files]
    except FileNotFoundError:
        print(f"❌ ERROR: The folder '{folder_path}' was not found.")
        return

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['filename', 'label'])
        for name in filenames:
            csv_writer.writerow([name, ''])

    print(f"✅ Success! Created '{csv_path}' with {len(filenames)} filenames.")


create_image_csv_by_date(image_folder_path, output_csv_file)