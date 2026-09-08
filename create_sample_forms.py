"""Create a sample Hindi GST Forms PDF for testing."""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Try to register a Hindi font if available
try:
    pdfmetrics.registerFont(TTFont('NotoSansDevanagari', '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf'))
    HAS_HINDI_FONT = True
except:
    HAS_HINDI_FONT = False

def create_sample_forms_pdf(output_path="data/forms/cgst_forms_hindi.pdf"):
    """Create a sample GST forms PDF with Hindi content."""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    c = canvas.Canvas(str(output_path), pagesize=A4)
    width, height = A4
    
    # Form 1: GST CMP-01
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, y, "GST CMP-01")
    y -= 25
    
    c.setFont("Helvetica", 12)
    c.drawString(72, y, "Application for Composition Levy")
    y -= 30
    
    c.setFont("Helvetica-Bold", 10)
    c.drawString(72, y, "Part I - Business Details")
    y -= 20
    
    c.setFont("Helvetica", 10)
    c.drawString(72, y, "1. Legal Name of the Registered Person:")
    y -= 15
    c.drawString(72, y, "2. GSTIN/UIN:")
    y -= 15
    c.drawString(72, y, "3. Principal Place of Business:")
    y -= 30
    
    c.setFont("Helvetica-Bold", 10)
    c.drawString(72, y, "Part II - Declaration")
    y -= 20
    
    c.setFont("Helvetica", 10)
    c.drawString(72, y, "4. I hereby declare that the information given above is true.")
    y -= 30
    
    c.drawString(72, y, "Instructions:")
    y -= 15
    c.drawString(72, y, "- Fill all fields in block letters")
    y -= 15
    c.drawString(72, y, "- Attach required documents")
    y -= 30
    
    c.drawString(72, y, "Verification:")
    y -= 15
    c.drawString(72, y, "I solemnly affirm that the details provided are correct.")
    y -= 30
    
    c.drawString(72, y, "Attachments Required:")
    y -= 15
    c.drawString(72, y, "- Proof of business registration")
    y -= 15
    c.drawString(72, y, "- Identity proof of authorized signatory")
    
    c.showPage()
    
    # Form 2: GST REG-01
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, y, "GST REG-01")
    y -= 25
    
    c.setFont("Helvetica", 12)
    c.drawString(72, y, "Application for New Registration")
    y -= 30
    
    c.setFont("Helvetica-Bold", 10)
    c.drawString(72, y, "Part A - Taxpayer Information")
    y -= 20
    
    c.setFont("Helvetica", 10)
    c.drawString(72, y, "1. Mobile Number:")
    y -= 15
    c.drawString(72, y, "2. Email Address:")
    y -= 15
    c.drawString(72, y, "3. State/UT:")
    y -= 30
    
    c.setFont("Helvetica-Bold", 10)
    c.drawString(72, y, "Part B - Business Details")
    y -= 20
    
    c.drawString(72, y, "4. Legal Name of Business:")
    y -= 15
    c.drawString(72, y, "5. Trade Name:")
    y -= 15
    c.drawString(72, y, "6. Constitution of Business:")
    y -= 30
    
    c.drawString(72, y, "Rule 8 of CGST Rules, 2017")
    y -= 30
    
    c.drawString(72, y, "Instructions:")
    y -= 15
    c.drawString(72, y, "- Submit within 30 days of becoming liable")
    y -= 15
    c.drawString(72, y, "- Upload supporting documents")
    
    c.showPage()
    
    # Form 3: GSTR-7
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, y, "GSTR-7")
    y -= 25
    
    c.setFont("Helvetica", 12)
    c.drawString(72, y, "Return for TDS Deductors")
    y -= 30
    
    c.setFont("Helvetica-Bold", 10)
    c.drawString(72, y, "Part 1 - GSTIN")
    y -= 20
    
    c.setFont("Helvetica", 10)
    c.drawString(72, y, "1. GSTIN of TDS Deductor:")
    y -= 15
    c.drawString(72, y, "2. Legal Name:")
    y -= 30
    
    c.setFont("Helvetica-Bold", 10)
    c.drawString(72, y, "Part 2 - TDS Details")
    y -= 20
    
    c.drawString(72, y, "3. TDS Certificate Details:")
    y -= 15
    c.drawString(72, y, "4. Amount Paid:")
    y -= 30
    
    c.drawString(72, y, "Verification:")
    y -= 15
    c.drawString(72, y, "This return is filed electronically under Rule 66.")
    
    c.showPage()
    
    # Form 4: GST DRC-01
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, y, "GST DRC-01")
    y -= 25
    
    c.setFont("Helvetica", 12)
    c.drawString(72, y, "Show Cause Notice for Demand")
    y -= 30
    
    c.setFont("Helvetica-Bold", 10)
    c.drawString(72, y, "Part I - Details of Proceeding")
    y -= 20
    
    c.setFont("Helvetica", 10)
    c.drawString(72, y, "1. Reference Number:")
    y -= 15
    c.drawString(72, y, "2. Period:")
    y -= 15
    c.drawString(72, y, "3. Section violated:")
    y -= 30
    
    c.setFont("Helvetica-Bold", 10)
    c.drawString(72, y, "Part II - Grounds")
    y -= 20
    
    c.drawString(72, y, "4. Reasons for demand:")
    y -= 15
    c.drawString(72, y, "5. Amount demanded:")
    y -= 30
    
    c.drawString(72, y, "Rule 142 of CGST Rules")
    
    c.save()
    print(f"Created sample PDF at {output_path}")
    return output_path

if __name__ == "__main__":
    create_sample_forms_pdf()
