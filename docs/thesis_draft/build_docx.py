"""
Build the thesis Word file from the chapter Markdown files
==========================================================
Reads docs/thesis_draft/00_front.md ... 08_references.md and writes
docs/thesis_draft/THESIS.docx with:

- A4, margins 2.5 cm (left 3 cm for binding), Times New Roman 12 pt,
  1.5 line spacing, justified body text
- title page, declaration, abstract, table of contents, list of figures,
  list of tables, abbreviations (front matter numbered i, ii, iii, ...)
- chapters numbered from page 1, each chapter on a new page
- figure captions below figures, table captions above tables
- equations as native Word equations (LaTeX -> MathML -> OMML)

The table of contents and the lists are Word fields; Word asks to update
them when the file is opened (answer "Yes").

Usage:
    .venv/bin/python docs/thesis_draft/build_docx.py
"""

import copy
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from latex2mathml.converter import convert as tex2mml
from lxml import etree
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "THESIS_FINAL.docx"
XSL = Path("/Applications/Microsoft Word.app/Contents/Resources/mathml2omml.xsl")
MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"

TITLE = ("Integrated Edge-to-Cloud Systems: Federated Learning "
         "for Sensor Fusion and Object Recognition")
FONT = "Times New Roman"
BLACK = RGBColor(0, 0, 0)

_xslt = etree.XSLT(etree.parse(str(XSL)))
INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\$[^$]+\$)")


# ---------------------------------------------------------------- helpers

def omml(tex):
    mml = etree.fromstring(tex2mml(tex).encode())
    return copy.deepcopy(_xslt(mml).getroot())


def set_font(style, size, bold=None, italic=None, name=FONT):
    style.font.name = name
    style.font.size = Pt(size)
    style.font.color.rgb = BLACK
    if bold is not None:
        style.font.bold = bold
    if italic is not None:
        style.font.italic = italic
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), name)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        fonts.attrib.pop(qn(attr), None)


def add_field(paragraph, instr, placeholder=""):
    """Insert a Word field (TOC, PAGE, ...) into a paragraph."""
    def r(tag, **attrs):
        run = OxmlElement("w:r")
        el = OxmlElement(tag)
        for k, v in attrs.items():
            el.set(qn(k), v)
        run.append(el)
        return run, el
    begin, _ = r("w:fldChar", **{"w:fldCharType": "begin"})
    instr_run, it = r("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = instr
    sep, _ = r("w:fldChar", **{"w:fldCharType": "separate"})
    text_run = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = placeholder
    text_run.append(t)
    end, _ = r("w:fldChar", **{"w:fldCharType": "end"})
    for el in (begin, instr_run, sep, text_run, end):
        paragraph._p.append(el)


def page_numbers(section, fmt, start=None):
    sect = section._sectPr
    pg = sect.find(qn("w:pgNumType"))
    if pg is None:
        pg = OxmlElement("w:pgNumType")
        sect.append(pg)
    pg.set(qn("w:fmt"), fmt)
    if start is not None:
        pg.set(qn("w:start"), str(start))
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    add_field(p, "PAGE", "1")


def shade(cell, hex_fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcpr.append(shd)


def repeat_header(row):
    trpr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trpr.append(el)


def no_split(row):
    trpr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:cantSplit")
    el.set(qn("w:val"), "true")
    trpr.append(el)


# ---------------------------------------------------------------- document

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.top_margin = sec.bottom_margin = Cm(2.5)
sec.left_margin, sec.right_margin = Cm(3.0), Cm(2.5)
sec.different_first_page_header_footer = True     # no number on title page

styles = doc.styles
normal = styles["Normal"]
set_font(normal, 12)
normal.paragraph_format.line_spacing = 1.5
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

for name, size, before, after in (("Heading 1", 18, 0, 18), ("Heading 2", 14, 18, 6),
                                  ("Heading 3", 12, 12, 6)):
    st = styles[name]
    set_font(st, size, bold=True)
    pf = st.paragraph_format
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.keep_with_next = True
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.line_spacing = 1.15
styles["Heading 1"].paragraph_format.page_break_before = True


def new_style(name, size, italic=False, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER,
              spacing=1.0, before=3, after=12, font=FONT):
    st = styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    st.base_style = normal
    set_font(st, size, bold=bold, italic=italic, name=font)
    pf = st.paragraph_format
    pf.alignment, pf.line_spacing = align, spacing
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    return st


new_style("Figure Caption", 10, italic=True)
new_style("Table Caption", 10, italic=True, before=12, after=4).paragraph_format.keep_with_next = True
new_style("Listing Caption", 10, italic=True)
new_style("Source Note", 9, before=0, after=12)


def split_source(caption):
    """'Title. Source: x' -> ('Title.', 'Source: x')"""
    m = re.search(r"\s+Sources?:\s", caption)
    if not m:
        return caption, None
    return caption[:m.start()].strip(), caption[m.start():].strip()
front = new_style("Front Title", 18, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
                  spacing=1.15, before=0, after=18)
front.paragraph_format.page_break_before = True
front.paragraph_format.keep_with_next = True
code_st = new_style("Code", 9, align=WD_ALIGN_PARAGRAPH.LEFT, before=0, after=0,
                    font="Courier New")
new_style("Table Text", 9.5, align=WD_ALIGN_PARAGRAPH.LEFT, before=1, after=1)
ref_st = new_style("Reference", 11, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.15,
                   before=0, after=6)
ref_st.paragraph_format.left_indent = Cm(1.0)
ref_st.paragraph_format.first_line_indent = Cm(-1.0)

# Word updates the TOC / list fields when the file is opened
if "--no-update" not in sys.argv:
    upd = OxmlElement("w:updateFields")
    upd.set(qn("w:val"), "true")
    doc.settings.element.append(upd)


# ---------------------------------------------------------------- title page

def centered(text, size, bold=False, before=0, after=6, italic=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(before), Pt(after)
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(text)
    r.font.size, r.bold, r.italic = Pt(size), bold, italic
    return p


centered("Berliner Hochschule für Technik", 16, bold=True, before=24)
centered("Department VII", 12)
centered("M.Sc. Information and Communication Engineering", 12, after=60)
centered("Master's Thesis", 14, after=24)
centered(TITLE, 20, bold=True, after=60)
centered("submitted by", 12, italic=True)
centered("Adnan Anwar Rajput", 14, bold=True)
centered("Matriculation number 931043", 12, after=60)
centered("First supervisor: Prof. Dr. André Jakob", 12)
centered("Second supervisor: Prof. Dr. Marcus Purat", 12, after=60)
centered("Berlin, October 2026", 12)


# ---------------------------------------------------------------- markdown

def add_inline(p, text):
    text = text.replace("⟦", "").replace("⟧", "")
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**"):
            p.add_run(part[2:-2]).bold = True
        elif part.startswith("`"):
            r = p.add_run(part[1:-1])
            small = p.style.name in ("Source Note", "Figure Caption", "Table Caption", "Table Text")
            r.font.name, r.font.size = "Courier New", Pt(8.5 if small else 10)
        elif part.startswith("$"):
            try:
                p._p.append(omml(part[1:-1]))
            except Exception:
                p.add_run(part[1:-1]).italic = True
        elif part.startswith("*"):
            p.add_run(part[1:-1]).italic = True
        else:
            p.add_run(part)


def add_table(rows, source=None):
    rows = [r for r in rows if not re.match(r"^\|\s*:?-+", r)]
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    ncol = max(len(c) for c in cells)
    t = doc.add_table(rows=len(cells), cols=ncol)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(cells):
        no_split(t.rows[i])
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            para = cell.paragraphs[0]
            para.style = "Table Text"
            add_inline(para, val)
            if i == 0:
                for r in para.runs:
                    r.bold = True
                shade(cell, "E7E6E6")
    repeat_header(t.rows[0])
    if len(cells) <= 8:                      # keep short tables on one page
        for row in t.rows[:-1]:
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.paragraph_format.keep_with_next = True
    if source:
        note = doc.add_paragraph(style="Source Note")
        note.paragraph_format.space_before = Pt(3)
        add_inline(note, source)
    else:
        gap = doc.add_paragraph()
        gap.paragraph_format.space_after = Pt(0)
        gap.paragraph_format.line_spacing = 1.0


def add_picture(path, caption):
    full = REPO / path
    w, h = Image.open(full).size
    width = 15.0
    max_h = 10.0 if h > 1.6 * w else 14.0      # very tall, narrow diagrams
    if h / w * width > max_h:
        width = max_h * w / h
    doc.add_picture(str(full), width=Cm(width))
    pic = doc.paragraphs[-1]
    pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pic.paragraph_format.keep_with_next = True
    pic.paragraph_format.space_before = Pt(6)
    pic.paragraph_format.line_spacing = 1.0
    title, source = split_source(caption)
    cap = doc.add_paragraph(style="Figure Caption")
    add_inline(cap, title)
    if source:
        cap.paragraph_format.space_after = Pt(0)
        cap.paragraph_format.keep_with_next = True
        add_inline(doc.add_paragraph(style="Source Note"), source)


def insert_lists():
    p = doc.add_paragraph("Table of Contents", style="Front Title")
    add_field(doc.add_paragraph(), 'TOC \\o "1-3" \\h \\z \\u',
              "Right-click here and choose Update Field.")
    doc.add_paragraph("List of Figures", style="Front Title")
    add_field(doc.add_paragraph(), 'TOC \\h \\z \\t "Figure Caption,1"',
              "Right-click here and choose Update Field.")
    doc.add_paragraph("List of Tables", style="Front Title")
    add_field(doc.add_paragraph(), 'TOC \\h \\z \\t "Table Caption,1"',
              "Right-click here and choose Update Field.")
    return p


def new_numbering():
    """A fresh numbering instance so every numbered list starts at 1."""
    style = styles["List Number"]
    num_id = style.element.pPr.numPr.numId.val
    numbering = doc.part.numbering_part.element
    abstract_id = numbering.num_having_numId(num_id).abstractNumId.val
    num = numbering.add_num(abstract_id)
    override = num.add_lvlOverride(ilvl=0)
    override.add_startOverride(1)
    return num.numId


def set_num(paragraph, num_id):
    ppr = paragraph._p.get_or_add_pPr()
    numpr = ppr.get_or_add_numPr()
    numpr.get_or_add_ilvl().val = 0
    numpr.get_or_add_numId().val = num_id


def convert(md_path, is_front=False):
    is_refs = md_path.name.startswith("08")
    list_num = None
    lines = md_path.read_text().splitlines()
    i, para = 0, []

    def flush():
        if para:
            add_inline(doc.add_paragraph(), " ".join(s.strip() for s in para))
            para.clear()

    def caption_block(start):
        """Read a *...* caption that may span several lines."""
        text, j = lines[start].strip(), start + 1
        while not text.endswith("*") and j < len(lines) and lines[j].strip():
            text += " " + lines[j].strip()
            j += 1
        return text.strip("*").strip(), j

    while i < len(lines):
        ln = lines[i]

        if ln.startswith("|"):                                 # table
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            j = i
            while j < len(lines) and not lines[j].strip():
                j += 1
            source = None
            if j < len(lines) and lines[j].startswith("*Table"):
                cap, i = caption_block(j)
                cap, source = split_source(cap)
                add_inline(doc.add_paragraph(style="Table Caption"), cap)
            add_table(rows, source)
            continue

        if ln.startswith("```"):                               # code block
            flush()
            i += 1
            block = []
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            for k, cl in enumerate(block):
                p = doc.add_paragraph(cl if cl else " ", style="Code")
                p.paragraph_format.keep_with_next = k < min(len(block) - 1, 12)
                p.paragraph_format.left_indent = Cm(0.5)
            if block:
                p.paragraph_format.space_after = Pt(8)
            continue

        if ln.startswith("*Listing"):
            flush()
            cap, i = caption_block(i)
            add_inline(doc.add_paragraph(style="Listing Caption"), cap)
            continue

        mi = re.match(r"^!\[(.*)\]\((.*)\)\s*$", ln)           # figure
        if mi:
            flush()
            add_picture(mi.group(2), mi.group(1))
            i += 1
            continue

        if ln.startswith("$$"):                                # display equation
            flush()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            mp = etree.SubElement(p._p, f"{{{MATH_NS}}}oMathPara")
            mp.append(omml(ln.strip().strip("$")))
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)", ln)                   # heading
        if m:
            flush()
            level, text = len(m.group(1)), m.group(2)
            if is_front:
                if text == "List of Abbreviations":
                    insert_lists()
                doc.add_paragraph(text, style="Front Title")
            else:
                add_inline(doc.add_heading(level=min(level, 3)), text)
            i += 1
            continue

        if re.match(r"^\s*([-*]|\d+\.)\s+", ln):                 # list item
            flush()
            numbered = bool(re.match(r"^\s*\d+\.", ln))
            if numbered and (list_num is None or re.match(r"^\s*1\.", ln)):
                list_num = new_numbering()
            text = re.sub(r"^\s*([-*]|\d+\.)\s+", "", ln)
            i += 1
            while (i < len(lines) and lines[i].startswith("  ") and lines[i].strip()
                   and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[i])
                   and not lines[i].strip().startswith("$$")):
                text += " " + lines[i].strip()
                i += 1
            p = doc.add_paragraph(style="List Number" if numbered else "List Bullet")
            if numbered:
                set_num(p, list_num)
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(3)
            add_inline(p, text)
            continue

        if ln.startswith(">"):                                 # quote / note
            flush()
            text = ln.lstrip("> ")
            i += 1
            while i < len(lines) and lines[i].startswith(">"):
                text += " " + lines[i].lstrip("> ")
                i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = p.paragraph_format.right_indent = Cm(1)
            add_inline(p, text)
            continue

        if is_refs and re.match(r"^\[\d+\] [A-Z]", ln):        # reference entry
            flush()
            text = ln
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^\[\d+\] [A-Z]", lines[i]):
                text += " " + lines[i].strip()
                i += 1
            add_inline(doc.add_paragraph(style="Reference"), text)
            continue

        if not ln.strip() or ln.strip() == "---":
            flush()
            i += 1
            continue

        para.append(ln)
        i += 1
    flush()


chapters = sorted(HERE.glob("0*.md"))
convert(chapters[0], is_front=True)                # declaration ... abbreviations
page_numbers(doc.sections[0], "lowerRoman")

main = doc.add_section(WD_SECTION.NEW_PAGE)
main.different_first_page_header_footer = False
page_numbers(main, "decimal", start=1)
for md in chapters[1:]:
    convert(md)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
out = Path(args[0]) if args else OUT
doc.save(out)
print("wrote", out)
