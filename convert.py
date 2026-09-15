import os
import markdown
from weasyprint import HTML, CSS

os.makedirs('docs-pdf', exist_ok=True)
md_files = [f for f in os.listdir('docs') if f.endswith('.md')]

css = CSS(string='''
    body { font-family: Arial, sans-serif; line-height: 1.6; margin: 2cm; }
    code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-family: monospace; }
    pre { background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }
    h1, h2, h3 { color: #333; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 1rem; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background-color: #f2f2f2; }
''')

for f in md_files:
    print(f"Converting {f}...")
    with open(f"docs/{f}", "r") as mdf:
        html = markdown.markdown(mdf.read(), extensions=['tables', 'fenced_code'])
        # Add basic HTML structure
        html = f"<html><body>{html}</body></html>"
        
    pdf_path = f"docs-pdf/{f.replace('.md', '.pdf')}"
    HTML(string=html, base_url='docs').write_pdf(pdf_path, stylesheets=[css])
    print(f"Saved to {pdf_path}")
