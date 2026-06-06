"""PDF report generators: per-embryo + PT certificate."""
from datetime import datetime, timedelta
import tempfile
from fpdf import FPDF


def _tmp(img):
    f = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    img.save(f.name); return f.name


def build_pdf_report(img, r, patient='', embryo='', model='Blast-YOLO'):
    pdf = FPDF('P','mm','A4'); pdf.add_page()
    pdf.set_font('Helvetica','B',16)
    pdf.cell(0,10,'Blastocyst Grading Report - RESEARCH USE ONLY',
             ln=True, align='C')
    pdf.set_font('Helvetica','',9)
    pdf.cell(0,6, f"Generated {datetime.now():%Y-%m-%d %H:%M} . Model: {model}",
             ln=True, align='C')
    pdf.ln(4)
    pdf.set_font('Helvetica','B',11); pdf.cell(45,7,'Patient / Cycle:')
    pdf.set_font('Helvetica','',11); pdf.cell(0,7, patient or '-', ln=True)
    pdf.set_font('Helvetica','B',11); pdf.cell(45,7,'Embryo ID:')
    pdf.set_font('Helvetica','',11); pdf.cell(0,7, embryo or '-', ln=True)
    pdf.ln(2)
    pdf.set_fill_color(240,244,252); pdf.rect(10, pdf.get_y(), 190, 22, 'F')
    pdf.set_xy(12, pdf.get_y()+4)
    pdf.set_font('Helvetica','B',22); pdf.cell(60,10, r['grade'])
    pdf.set_font('Helvetica','',10)
    pdf.cell(0,6, f"Confidence {r['confidence']*100:.1f}%  .  "
                  f"Clinical group: {r['clinical_group']}", ln=True)
    if r.get('action'):
        pdf.set_font('Helvetica','I',9)
        pdf.cell(0,5, f"Action: {r['action']}", ln=True)
    pdf.ln(18)
    pdf.set_font('Helvetica','B',11); pdf.cell(0,7,'Top-3 alternatives', ln=True)
    pdf.set_font('Helvetica','',10)
    for g,p in r['top3']:
        pdf.cell(0,6, f"  {g:<6}   {p*100:5.1f} %", ln=True)
    pdf.ln(2)
    a = _tmp(img.resize((400,400)))
    pdf.image(a, x=15, y=pdf.get_y(), w=85)
    if r.get('overlay') is not None:
        b = _tmp(r['overlay'].resize((400,400)))
        pdf.image(b, x=110, y=pdf.get_y(), w=85)
    pdf.ln(90)
    pdf.set_font('Helvetica','I',8); pdf.set_text_color(120,120,120)
    pdf.multi_cell(0,4,
        'RESEARCH USE ONLY. Decision-support tool. The trained embryologist '
        'remains responsible for the clinical grade and downstream care. '
        'Inference performed locally; image data did not leave this device.')
    return bytes(pdf.output(dest='S'))


def build_pt_certificate(examinee, total, ex_score, ai_score,
                         elapsed: timedelta, model='Blast-YOLO'):
    """One-page certificate-style PT report."""
    pdf = FPDF('L','mm','A4'); pdf.add_page()
    pdf.set_font('Helvetica','B',24)
    pdf.cell(0,18,'Proficiency Test Report', ln=True, align='C')
    pdf.set_font('Helvetica','',12)
    pdf.cell(0,8,'Blastocyst Grading - RESEARCH USE ONLY',
             ln=True, align='C')
    pdf.ln(8)
    pdf.set_font('Helvetica','',13)
    pdf.cell(0,9, f"Examinee: {examinee}", ln=True, align='C')
    pdf.cell(0,9, f"Date: {datetime.now():%d %B %Y, %H:%M}",
             ln=True, align='C')
    pdf.ln(10)

    pct_ex = ex_score/total*100; pct_ai = ai_score/total*100
    pdf.set_font('Helvetica','B',16); pdf.cell(0,10,'Results',
                                               ln=True, align='C')
    pdf.set_font('Helvetica','',13)
    pdf.cell(0,8, f"Total questions: {total}", ln=True, align='C')
    pdf.cell(0,8, f"Examinee score : {ex_score} / {total}  ({pct_ex:.1f}%)",
             ln=True, align='C')
    pdf.cell(0,8, f"AI score       : {ai_score} / {total}  ({pct_ai:.1f}%)",
             ln=True, align='C')
    mins = elapsed.seconds//60; secs = elapsed.seconds%60
    pdf.cell(0,8, f"Time taken     : {mins} min {secs} s",
             ln=True, align='C')
    pdf.ln(6)
    pdf.set_font('Helvetica','I',11)
    band = 'Excellent' if pct_ex>=85 else ('Good' if pct_ex>=70
            else ('Acceptable' if pct_ex>=50 else 'Needs review'))
    pdf.cell(0,8, f"Performance band: {band}", ln=True, align='C')
    pdf.ln(20)
    pdf.set_font('Helvetica','',10)
    pdf.cell(0,6, f"Model used for AI comparison: {model}",
             ln=True, align='C')
    pdf.ln(15)
    pdf.set_font('Helvetica','I',8); pdf.set_text_color(120,120,120)
    pdf.multi_cell(0,4,
        'RESEARCH USE ONLY. This Proficiency Test report is for '
        'educational and quality-assurance purposes within the laboratory; '
        'it does not constitute regulatory certification. The AI score is '
        'reported as a co-observer reference, not as a ground truth. '
        'Ground truth for each question is the original radiologist / '
        'embryologist label embedded in the filename of the test image.')
    return bytes(pdf.output(dest='S'))
