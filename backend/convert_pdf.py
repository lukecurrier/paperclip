import argparse
import os
from pathlib import Path
import uuid

def create_paper_directory(paper_id):
    """
    Create a directory for the paper in the papers/ directory
    """
    base_dir = Path(__file__).parent / 'papers'
    paper_dir = base_dir / paper_id
    paper_dir.mkdir(parents=True, exist_ok=True)
    return paper_dir

def convert_pdf_to_markdown(pdf_path, paper_id):
    """
    Convert a PDF to markdown and save it in the papers directory
    """
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
    except ImportError:
        print("Make sure Marker is installed!")
        return False
    
    if not os.path.isfile(pdf_path):
        print(f"'{pdf_path}' not found.")
        return False
    
    try:
        converter = PdfConverter(
            artifact_dict=create_model_dict(),
        )
        
        print(f"Converting {pdf_path}...")
        rendered = converter(pdf_path)

        # Create paper directory
        paper_dir = create_paper_directory(paper_id)
        
        # Save the markdown file
        md_path = paper_dir / f"{paper_id}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(rendered.markdown)
            
        print(f"Successfully converted to {md_path}")
        return True
        
    except Exception as e:
        print(f"Error converting PDF: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert PDF to markdown and save in papers directory')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('paper_id', help='ID of the paper')
    
    args = parser.parse_args()
    
    success = convert_pdf_to_markdown(args.pdf_path, args.paper_id)
    if success:
        print("Conversion completed successfully!")
    else:
        print("Conversion failed.")

if __name__ == "__main__":
    main()