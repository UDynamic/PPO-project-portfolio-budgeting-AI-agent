#!/usr/bin/env python3
"""
tex_to_persian_docx.py
======================
Convert English .tex files to a Persian DOCX conforming to the
IUST Industrial Engineering thesis template (final-rev07.docx).

Usage:
    python tex_to_persian_docx.py input.tex --title "عنوان پایان‌نامه" --output output.docx

    # With all metadata:
    python tex_to_persian_docx.py chapters/*.tex \
        --title "عنوان پایان‌نامه" \
        --author "نام دانشجو" \
        --student-id "402XXXXXX" \
        --supervisor "دکتر نام استاد" \
        --field "مهندسی صنایع" \
        --major "مدیریت مهندسی" \
        --date "شهریور 1405" \
        --translate \
        --output thesis_persian.docx

Requirements:
    pip install python-docx anthropic
    (optional for translation) anthropic>=0.25

Template spec extracted from final-rev07.docx (IUST IE Department).
"""

import re
import sys
import argparse
import os
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# TEMPLATE CONSTANTS  (from final-rev07.docx extraction)
# ---------------------------------------------------------------------------

# Page layout (in EMU: 1 twip = 635 EMU; 1 cm = 360000 EMU)
PAGE_WIDTH_EMU  = 11906 * 635   # A4
PAGE_HEIGHT_EMU = 16838 * 635
MARGIN_TOP_EMU    = 1701 * 635  # 3.0 cm
MARGIN_BOTTOM_EMU = 1701 * 635  # 3.0 cm
MARGIN_LEFT_EMU   = 1701 * 635  # 3.0 cm  (left = outer for RTL)
MARGIN_RIGHT_EMU  = 2268 * 635  # 4.0 cm  (right = binding side in RTL)
HEADER_EMU        = 851  * 635
FOOTER_EMU        = 1021 * 635

# Fonts
FONT_BODY_CS      = "Zar"            # Persian body text
FONT_HEADING_CS   = "B Nazanin"      # chapter/section headings
FONT_TITLE_CS     = "Nazanin"        # cover page title
FONT_LATIN        = "Times New Roman"
FONT_CAPTION_CS   = "Zar"
FONT_PAGENO_CS    = "Zar"

# Sizes in half-points (docx unit)
SIZE_BODY_CS      = 28   # 14 pt
SIZE_HEADING      = 28   # 14 pt
SIZE_TITLE_CS     = 36   # 18 pt
SIZE_SUBTITLE_CS  = 32   # 16 pt
SIZE_CAPTION_CS   = 20   # 10 pt
SIZE_PAGENO_CS    = 26   # 13 pt
SIZE_TOC_CS       = 20   # 10 pt

# Line spacing  (288/240 = 1.2× single)
LINE_SPACING_BODY = 288   # auto rule


# ---------------------------------------------------------------------------
# LATEX PARSER  — naive but handles the common thesis macros
# ---------------------------------------------------------------------------

def parse_tex_file(path: str) -> list[dict]:
    """
    Parse a .tex file into a list of blocks:
      {'type': 'chapter'|'section'|'subsection'|'subsubsection'|
               'paragraph'|'figure_caption'|'table_caption'|
               'equation'|'itemize'|'enumerate'|'verbatim',
       'content': str,
       'items': [str, ...]   # for list types
      }
    """
    with open(path, encoding='utf-8') as f:
        raw = f.read()

    blocks = []
    lines  = raw.split('\n')
    i      = 0

    def clean(s: str) -> str:
        """Strip LaTeX markup for plain text."""
        # Remove common LaTeX commands
        s = re.sub(r'\\emph\{([^}]*)\}',       r'\1', s)
        s = re.sub(r'\\textbf\{([^}]*)\}',     r'\1', s)
        s = re.sub(r'\\textit\{([^}]*)\}',     r'\1', s)
        s = re.sub(r'\\cite\{[^}]*\}',         '[?]', s)
        s = re.sub(r'\\ref\{[^}]*\}',          '[?]', s)
        s = re.sub(r'\\label\{[^}]*\}',        '',    s)
        s = re.sub(r'\\footnote\{[^}]*\}',     '',    s)
        s = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', s)
        s = re.sub(r'\\[a-zA-Z]+\b',           '',    s)
        s = re.sub(r'%.*',                      '',    s)
        s = s.replace('~', '\u00a0').replace('---', '—').replace('--', '–')
        return s.strip()

    def collect_brace(text: str) -> str:
        """Extract {content} after a command."""
        m = re.match(r'\{([^}]*)\}', text)
        return m.group(1) if m else ''

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip preamble / empty / comments
        if not stripped or stripped.startswith('%'):
            i += 1
            continue

        # Document begin/end
        if '\\begin{document}' in stripped or '\\end{document}' in stripped:
            i += 1
            continue

        # Chapter
        m = re.match(r'\\chapter\*?\{(.*)\}', stripped)
        if m:
            blocks.append({'type': 'chapter', 'content': clean(m.group(1))})
            i += 1
            continue

        # Section
        m = re.match(r'\\section\*?\{(.*)\}', stripped)
        if m:
            blocks.append({'type': 'section', 'content': clean(m.group(1))})
            i += 1
            continue

        # Subsection
        m = re.match(r'\\subsection\*?\{(.*)\}', stripped)
        if m:
            blocks.append({'type': 'subsection', 'content': clean(m.group(1))})
            i += 1
            continue

        # Subsubsection
        m = re.match(r'\\subsubsection\*?\{(.*)\}', stripped)
        if m:
            blocks.append({'type': 'subsubsection', 'content': clean(m.group(1))})
            i += 1
            continue

        # Figure / table captions — grab from \caption{}
        m = re.match(r'\\caption\{(.*)\}', stripped)
        if m:
            blocks.append({'type': 'figure_caption', 'content': clean(m.group(1))})
            i += 1
            continue

        # Equation blocks
        if '\\begin{equation}' in stripped or '\\begin{align}' in stripped:
            eq_lines = []
            while i < len(lines):
                eq_lines.append(lines[i])
                if '\\end{equation}' in lines[i] or '\\end{align}' in lines[i]:
                    break
                i += 1
            blocks.append({'type': 'equation', 'content': ' '.join(eq_lines)})
            i += 1
            continue

        # Verbatim
        if '\\begin{verbatim}' in stripped or '\\begin{lstlisting}' in stripped:
            v_lines = []
            while i < len(lines):
                if '\\end{verbatim}' in lines[i] or '\\end{lstlisting}' in lines[i]:
                    break
                v_lines.append(lines[i])
                i += 1
            blocks.append({'type': 'verbatim', 'content': '\n'.join(v_lines[1:])})
            i += 1
            continue

        # Itemize / enumerate
        if '\\begin{itemize}' in stripped or '\\begin{enumerate}' in stripped:
            list_type = 'itemize' if 'itemize' in stripped else 'enumerate'
            items = []
            i += 1
            while i < len(lines) and '\\end{' not in lines[i]:
                m2 = re.match(r'\s*\\item\s*(.*)', lines[i])
                if m2:
                    items.append(clean(m2.group(1)))
                i += 1
            blocks.append({'type': list_type, 'items': items, 'content': ''})
            i += 1
            continue

        # Skip other \begin{...} blocks (figure, table, etc.) — keep captions
        if re.match(r'\\begin\{', stripped):
            while i < len(lines) and '\\end{' not in lines[i]:
                # still capture \caption inside
                m_cap = re.match(r'\s*\\caption\{(.*)\}', lines[i])
                if m_cap:
                    blocks.append({'type': 'figure_caption', 'content': clean(m_cap.group(1))})
                i += 1
            i += 1
            continue

        # Regular paragraph text — collect until blank line or next command
        if stripped and not stripped.startswith('\\'):
            para_lines = []
            while i < len(lines):
                l = lines[i].strip()
                if not l or l.startswith('\\') or l.startswith('%'):
                    break
                para_lines.append(l)
                i += 1
            text = ' '.join(para_lines)
            text = clean(text)
            if text:
                blocks.append({'type': 'paragraph', 'content': text})
            continue

        i += 1

    return blocks


# ---------------------------------------------------------------------------
# OPTIONAL TRANSLATION via Anthropic API
# ---------------------------------------------------------------------------

def translate_blocks(blocks: list[dict]) -> list[dict]:
    """
    Translate English text in each block to Persian using the Anthropic API.
    Requires ANTHROPIC_API_KEY in env.
    """
    try:
        import anthropic
    except ImportError:
        print("[WARNING] anthropic package not installed. Skipping translation.", file=sys.stderr)
        print("          Install with:  pip install anthropic", file=sys.stderr)
        return blocks

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[WARNING] ANTHROPIC_API_KEY not set. Skipping translation.", file=sys.stderr)
        return blocks

    client = anthropic.Anthropic(api_key=api_key)

    system_prompt = """You are a professional academic translator specializing in Industrial Engineering and Operations Research.
Translate the given English academic text into formal Persian (Farsi) suitable for a university thesis.
Rules:
- Use formal academic Persian
- Keep technical terms in English where standard in Iranian academia (e.g., Reinforcement Learning, Markov Decision Process, NPV, etc.)
- Do NOT translate proper nouns, abbreviations, or mathematical notation
- Return ONLY the translated Persian text, nothing else
- Preserve paragraph structure"""

    translated = []
    for block in blocks:
        if block['type'] in ('chapter', 'section', 'subsection', 'subsubsection',
                              'paragraph', 'figure_caption', 'table_caption'):
            if not block['content'].strip():
                translated.append(block)
                continue
            print(f"  [translate] {block['type']}: {block['content'][:60]}…", file=sys.stderr)
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1024,
                    system=system_prompt,
                    messages=[{"role": "user", "content": block['content']}]
                )
                fa_text = response.content[0].text.strip()
                new_block = dict(block)
                new_block['content'] = fa_text
                new_block['original'] = block['content']
                translated.append(new_block)
            except Exception as e:
                print(f"  [WARNING] Translation failed: {e}", file=sys.stderr)
                translated.append(block)

        elif block['type'] in ('itemize', 'enumerate'):
            new_items = []
            for item in block.get('items', []):
                try:
                    response = client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=512,
                        system=system_prompt,
                        messages=[{"role": "user", "content": item}]
                    )
                    new_items.append(response.content[0].text.strip())
                except Exception:
                    new_items.append(item)
            new_block = dict(block)
            new_block['items'] = new_items
            translated.append(new_block)
        else:
            translated.append(block)

    return translated


# ---------------------------------------------------------------------------
# DOCX BUILDER
# ---------------------------------------------------------------------------

def build_docx(blocks: list[dict], output_path: str, meta: dict) -> None:
    """
    Build a DOCX matching the IUST thesis template spec.
    """
    try:
        from docx import Document
        from docx.shared import Pt, Cm, RGBColor, Emu
        from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
        import copy
    except ImportError:
        print("[ERROR] python-docx not installed. Run: pip install python-docx", file=sys.stderr)
        sys.exit(1)

    doc = Document()

    # ------------------------------------------------------------------
    # Page layout
    # ------------------------------------------------------------------
    section = doc.sections[0]
    section.page_width  = Emu(PAGE_WIDTH_EMU)
    section.page_height = Emu(PAGE_HEIGHT_EMU)
    section.top_margin    = Cm(3.0)
    section.bottom_margin = Cm(3.0)
    section.left_margin   = Cm(3.0)   # outer margin (RTL)
    section.right_margin  = Cm(4.0)   # binding margin (RTL)
    section.header_distance = Emu(HEADER_EMU)
    section.footer_distance = Emu(FOOTER_EMU)

    # Set section as RTL + RTL gutter
    sectPr = section._sectPr
    bidi_el = OxmlElement('w:bidi')
    sectPr.append(bidi_el)
    rtlGutter = OxmlElement('w:rtlGutter')
    sectPr.append(rtlGutter)

    # Page numbering — arabicAbjad for front matter
    pgNumType = OxmlElement('w:pgNumType')
    pgNumType.set(qn('w:fmt'), 'arabicAbjad')
    sectPr.append(pgNumType)

    # ------------------------------------------------------------------
    # Helper: apply Persian run formatting
    # ------------------------------------------------------------------
    def set_persian_run(run, size_half_pt: int, bold: bool = False,
                        font_cs: str = FONT_BODY_CS,
                        font_latin: str = FONT_LATIN):
        rPr = run._r.get_or_add_rPr()

        # Fonts
        rFonts = OxmlElement('w:rFonts')
        rFonts.set(qn('w:ascii'),     font_latin)
        rFonts.set(qn('w:hAnsi'),     font_latin)
        rFonts.set(qn('w:eastAsia'),  font_latin)
        rFonts.set(qn('w:cs'),        font_cs)
        rPr.append(rFonts)

        # Size
        sz = OxmlElement('w:sz')
        sz.set(qn('w:val'), str(size_half_pt))
        rPr.append(sz)
        szCs = OxmlElement('w:szCs')
        szCs.set(qn('w:val'), str(size_half_pt))
        rPr.append(szCs)

        if bold:
            rPr.append(OxmlElement('w:b'))
            rPr.append(OxmlElement('w:bCs'))

        # RTL run
        rPr.append(OxmlElement('w:rtl'))

        # Language
        lang = OxmlElement('w:lang')
        lang.set(qn('w:bidi'), 'fa-IR')
        rPr.append(lang)

    def set_persian_para(para, jc: str = 'lowKashida', bidi: bool = True,
                         line_spacing: int = LINE_SPACING_BODY):
        pPr = para._p.get_or_add_pPr()

        if bidi:
            pPr.append(OxmlElement('w:bidi'))

        jc_el = OxmlElement('w:jc')
        jc_el.set(qn('w:val'), jc)
        pPr.append(jc_el)

        spacing = OxmlElement('w:spacing')
        spacing.set(qn('w:after'),    '0')
        spacing.set(qn('w:line'),     str(line_spacing))
        spacing.set(qn('w:lineRule'), 'auto')
        pPr.append(spacing)

    # ------------------------------------------------------------------
    # Helper: add a heading paragraph
    # ------------------------------------------------------------------
    def add_heading(text: str, level: int) -> None:
        """level 1=chapter, 2=section, 3=subsection, 4=subsubsection"""
        sizes = {1: SIZE_HEADING, 2: SIZE_HEADING, 3: SIZE_HEADING, 4: SIZE_HEADING}
        sz = sizes.get(level, SIZE_BODY_CS)

        para = doc.add_paragraph()
        set_persian_para(para, jc='right', bidi=True, line_spacing=240)

        # Indent by level
        pPr = para._p.get_or_add_pPr()
        if level > 1:
            ind = OxmlElement('w:ind')
            ind.set(qn('w:right'), str((level - 1) * 360))
            pPr.append(ind)

        # Outline level for TOC
        if level <= 3:
            outlineLvl = OxmlElement('w:outlineLvl')
            outlineLvl.set(qn('w:val'), str(level - 1))
            pPr.append(outlineLvl)

        run = para.add_run(text)
        set_persian_run(run, size_half_pt=sz, bold=True,
                        font_cs=FONT_HEADING_CS)

    # ------------------------------------------------------------------
    # Helper: add body paragraph
    # ------------------------------------------------------------------
    def add_body_para(text: str) -> None:
        para = doc.add_paragraph()
        set_persian_para(para, jc='lowKashida', bidi=True,
                         line_spacing=LINE_SPACING_BODY)
        run = para.add_run(text)
        set_persian_run(run, size_half_pt=SIZE_BODY_CS,
                        font_cs=FONT_BODY_CS)

    # ------------------------------------------------------------------
    # Helper: add figure/table caption
    # ------------------------------------------------------------------
    def add_caption(text: str) -> None:
        para = doc.add_paragraph()
        pPr = para._p.get_or_add_pPr()
        jc_el = OxmlElement('w:jc')
        jc_el.set(qn('w:val'), 'center')
        pPr.append(jc_el)
        spacing = OxmlElement('w:spacing')
        spacing.set(qn('w:line'),     '240')
        spacing.set(qn('w:lineRule'), 'auto')
        pPr.append(spacing)
        run = para.add_run(text)
        set_persian_run(run, size_half_pt=SIZE_CAPTION_CS)

    # ------------------------------------------------------------------
    # Helper: add list
    # ------------------------------------------------------------------
    def add_list(items: list[str], ordered: bool = False) -> None:
        for item in items:
            para = doc.add_paragraph(style='List Bullet' if not ordered else 'List Number')
            set_persian_para(para, jc='right', bidi=True, line_spacing=240)
            run = para.add_run(item)
            set_persian_run(run, size_half_pt=SIZE_BODY_CS)

    # ------------------------------------------------------------------
    # Cover page (if metadata provided)
    # ------------------------------------------------------------------
    if meta.get('title'):
        # Title
        para = doc.add_paragraph()
        pPr = para._p.get_or_add_pPr()
        jc_el = OxmlElement('w:jc')
        jc_el.set(qn('w:val'), 'center')
        pPr.append(jc_el)
        pPr.append(OxmlElement('w:bidi'))
        spacing = OxmlElement('w:spacing')
        spacing.set(qn('w:after'), '300')
        pPr.append(spacing)

        run = para.add_run(meta['title'])
        set_persian_run(run, size_half_pt=SIZE_TITLE_CS, bold=True,
                        font_cs=FONT_TITLE_CS)

        # Metadata lines
        meta_lines = []
        if meta.get('author'):
            meta_lines.append(f"دانشجو: {meta['author']}")
        if meta.get('student_id'):
            meta_lines.append(meta['student_id'])
        if meta.get('supervisor'):
            meta_lines.append(f"استاد راهنما: {meta['supervisor']}")
        if meta.get('field'):
            meta_lines.append(f"رشته: {meta['field']}")
        if meta.get('major'):
            meta_lines.append(f"گرایش: {meta['major']}")
        if meta.get('date'):
            meta_lines.append(meta['date'])

        for line in meta_lines:
            para = doc.add_paragraph()
            set_persian_para(para, jc='center', bidi=True, line_spacing=240)
            run = para.add_run(line)
            set_persian_run(run, size_half_pt=SIZE_SUBTITLE_CS, bold=True,
                            font_cs=FONT_HEADING_CS)

        doc.add_page_break()

    # ------------------------------------------------------------------
    # Main content
    # ------------------------------------------------------------------
    chapter_num = 0
    section_num = 0
    subsection_num = 0

    for block in blocks:
        btype   = block['type']
        content = block.get('content', '')

        if btype == 'chapter':
            chapter_num += 1
            section_num = 0
            subsection_num = 0
            label = f"فصل {chapter_num}: {content}"
            add_heading(label, level=1)

        elif btype == 'section':
            section_num += 1
            subsection_num = 0
            label = f"{chapter_num}-{section_num}  {content}"
            add_heading(label, level=2)

        elif btype == 'subsection':
            subsection_num += 1
            label = f"{chapter_num}-{section_num}.{subsection_num}  {content}"
            add_heading(label, level=3)

        elif btype == 'subsubsection':
            add_heading(content, level=4)

        elif btype == 'paragraph':
            add_body_para(content)

        elif btype == 'figure_caption':
            add_caption(content)

        elif btype == 'table_caption':
            add_caption(content)

        elif btype == 'equation':
            # equations: centered, keep as-is (user can insert MathML later)
            para = doc.add_paragraph()
            pPr = para._p.get_or_add_pPr()
            jc_el = OxmlElement('w:jc')
            jc_el.set(qn('w:val'), 'center')
            pPr.append(jc_el)
            run = para.add_run(f"[معادله: {content[:80]}]")
            set_persian_run(run, size_half_pt=SIZE_BODY_CS)

        elif btype == 'verbatim':
            para = doc.add_paragraph()
            run = para.add_run(content)
            run.font.name = 'Courier New'
            run.font.size = Pt(10)

        elif btype == 'itemize':
            add_list(block.get('items', []), ordered=False)

        elif btype == 'enumerate':
            add_list(block.get('items', []), ordered=True)

    # ------------------------------------------------------------------
    # Footer with page number
    # ------------------------------------------------------------------
    footer = section.footer
    footer_para = footer.paragraphs[0]
    pPr = footer_para._p.get_or_add_pPr()
    jc_el = OxmlElement('w:jc')
    jc_el.set(qn('w:val'), 'center')
    pPr.append(jc_el)

    run = footer_para.add_run()
    rPr = run._r.get_or_add_rPr()
    rFonts = OxmlElement('w:rFonts')
    rFonts.set(qn('w:ascii'), FONT_LATIN)
    rFonts.set(qn('w:cs'),    FONT_PAGENO_CS)
    rPr.append(rFonts)
    sz_el = OxmlElement('w:szCs')
    sz_el.set(qn('w:val'), str(SIZE_PAGENO_CS))
    rPr.append(sz_el)

    # Insert PAGE field
    fldChar1 = OxmlElement('w:fldChar')
    fldChar1.set(qn('w:fldCharType'), 'begin')
    instrText = OxmlElement('w:instrText')
    instrText.text = ' PAGE  '
    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'end')
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    doc.save(output_path)
    print(f"[OK] Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Convert English .tex thesis to Persian DOCX (IUST IE template)')
    parser.add_argument('inputs', nargs='+', help='.tex input files')
    parser.add_argument('--output', '-o', default='thesis_persian.docx')
    parser.add_argument('--translate', action='store_true',
                        help='Translate English content to Persian via Anthropic API')
    # Metadata
    parser.add_argument('--title',      default='', help='Persian thesis title')
    parser.add_argument('--author',     default='', help='Student name (Persian)')
    parser.add_argument('--student-id', default='')
    parser.add_argument('--supervisor', default='')
    parser.add_argument('--field',      default='مهندسی صنایع')
    parser.add_argument('--major',      default='')
    parser.add_argument('--date',       default='')

    args = parser.parse_args()

    meta = {
        'title':      args.title,
        'author':     args.author,
        'student_id': args.student_id,
        'supervisor': args.supervisor,
        'field':      args.field,
        'major':      args.major,
        'date':       args.date,
    }

    # Parse all .tex files
    all_blocks = []
    for tex_path in args.inputs:
        print(f"[parse] {tex_path}")
        blocks = parse_tex_file(tex_path)
        all_blocks.extend(blocks)

    print(f"[info] {len(all_blocks)} blocks parsed.")

    # Translate if requested
    if args.translate:
        print("[translate] Translating with Anthropic API …")
        all_blocks = translate_blocks(all_blocks)

    # Build DOCX
    print(f"[build] Writing {args.output} …")
    build_docx(all_blocks, args.output, meta)


if __name__ == '__main__':
    main()
