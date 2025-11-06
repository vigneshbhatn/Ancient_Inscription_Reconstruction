import os
import csv

# --- CONFIGURATION ---
csv_with_filenames = r"D:\Major Project stuff\project_code\package\labels.csv"
image_folder_path = r"D:\Major Project stuff\project_code\package\FINAL_CLEAN_DATASET v2"
output_html_file = "labeling_tool.html"
# ---------------------

def create_labeling_tool(csv_path, img_folder, html_path):
    print("Step 2: Generating the HTML labeling tool...")
    html_content = """
    <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Image Labeling Tool</title><style>body{font-family:sans-serif;margin:2em;background-color:#f4f4f4}.container{max-width:800px;margin:auto;background:white;padding:20px;box-shadow:0 0 10px rgba(0,0,0,.1)}h1,h2{text-align:center}.item{display:flex;align-items:center;margin-bottom:15px;border-bottom:1px solid #eee;padding-bottom:15px}.item-index{font-size:1.1em;font-weight:700;color:#666;width:60px;text-align:right;margin-right:10px}.item img{width:100px;height:100px;object-fit:contain;margin-right:20px;border:1px solid #ccc}.item .filename{font-family:monospace;font-size:.8em;color:#555;display:block;margin-top:5px}.item input{width:100px;font-size:1.5em;text-align:center;padding:5px}.export-button{display:block;width:100%;padding:15px;font-size:1.2em;background-color:#28a745;color:#fff;border:none;cursor:pointer;text-align:center;margin-top:20px}.export-button:hover{background-color:#218838}</style></head><body><div class=container><h1>Image Labeling Tool</h1><div id=image-list>
    """
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            header = next(reader)
            for i, row in enumerate(reader, 1):
                filename = row[0]
                image_path = os.path.join(img_folder, filename).replace("\\", "/")
                html_content += f'<div class=item><div class=item-index>{i}.</div><img src="{image_path}" alt="{filename}"><div><input type=text data-filename="{filename}"><span class=filename>{filename}</span></div></div>'
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return
    html_content += """
            </div><button class=export-button onclick=exportCSV()>Export to CSV</button><h2>Instructions:</h2><p>1. Type the character label in the box next to each image.</p><p>2. When you are finished, click the 'Export to CSV' button.</p><p>3. A unique, timestamped CSV file will be downloaded.</p></div><script>function exportCSV(){let e="data:text/csv;charset=utf-8,\\uFEFFfilename,label\\n";const t=document.querySelectorAll(".item input");t.forEach(t=>{const n=t.getAttribute("data-filename"),a=t.value.replace(/,/g,"");a&&(e+=`${n},${a}\\n`)});const n=new Date,a=`${n.getFullYear()}-${String(n.getMonth()+1).padStart(2,"0")}-${String(n.getDate()).padStart(2,"0")}_${String(n.getHours()).padStart(2,"0")}-${String(n.getMinutes()).padStart(2,"0")}-${String(n.getSeconds()).padStart(2,"0")}`,l=`labels_${a}.csv`,o=encodeURI(e),s=document.createElement("a");s.setAttribute("href",o),s.setAttribute("download",l),document.body.appendChild(s),s.click()}</script></body></html>
    """
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"✅ Success! Your labeling tool has been created: {html_path}")

create_labeling_tool(csv_with_filenames, image_folder_path, output_html_file)