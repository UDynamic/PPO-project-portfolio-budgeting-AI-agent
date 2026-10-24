# IUST Persian Thesis Template — Complete Specification
Extracted from `final-rev07.docx` (Iran University of Science and Technology, Industrial Engineering)

---

## 1. Page Layout

| Property | Value |
|---|---|
| Paper size | A4 — 11906 × 16838 twips (21.0 × 29.7 cm) |
| Orientation | Portrait |
| Top margin | 1701 twips → **3.0 cm** |
| Bottom margin | 1701 twips → **3.0 cm** |
| Left margin | 1701 twips → **3.0 cm** |
| Right margin | 2268 twips → **4.0 cm** (binding side) |
| Header distance | 851 twips ≈ 1.5 cm |
| Footer distance | 1021 twips ≈ 1.8 cm |
| Gutter | 0 (RTL gutter is ON — binding is on right) |
| RTL gutter | **True** (Word moves the extra margin to the right side automatically) |
| Default tab stop | 720 twips (0.5 in / 1.27 cm) |

> **LaTeX equivalent:** A4, top/bottom 3 cm, left (inner for RTL) 3 cm, right (outer for RTL / binding) 4 cm.

---

## 2. Fonts

### 2.1 Persian / Complex-Script Fonts

| Usage | Font name |
|---|---|
| **Body text (متن)** | **Zar** (14 pt complex-script) |
| Headings (custom h1 char) | **B Nazanin** (bold) |
| Cover-page title (Title1) | **Nazanin** (18 pt cs, bold) |
| Section heading (Heading11) | **B Nazanin** implied (Bold, 14 pt) |
| TOC entries | **Times New Roman** (cs fallback for numbered TOC) |
| Page number footer | **Zar** (13 pt cs) |
| Normal paragraph fallback | **B Nazanin** (docDefaults cstheme = minorBidi) |

### 2.2 Latin / ASCII Fonts

| Usage | Font |
|---|---|
| Body / ASCII within Persian text | Times New Roman |
| Normal Web style | Times New Roman |
| Code / preformatted | Courier New |
| Theme major Latin | Aptos Display |
| Theme minor Latin | Aptos |

### 2.3 Theme Font Mappings
- `majorHAnsi` → **Aptos Display**; `majorBidi` (Arabic script) → **Times New Roman**
- `minorHAnsi` → **Aptos**; `minorBidi` → **Arial** (theme default, overridden in styles to B Nazanin / Zar)

---

## 3. Text Styles (Named Styles)

### 3.1 Body Text — Style `a` (متن)

| Property | Value |
|---|---|
| styleId | `a` |
| Display name | متن (body text) |
| Paragraph direction | **RTL (bidi = true)** |
| Justification | `lowKashida` (Persian full-justify with kashida extension) |
| Line spacing | `288` / `auto` ≈ **1.2× (about 20% above single)** |
| Space after | 0 pt |
| ASCII font | Times New Roman |
| Complex-script font | **Zar** |
| Complex-script size | **14 pt** (28 half-pts) |

### 3.2 Body Text Continuous — Style `a1` (متن پيوسته)

Same as `a` but `bidi=false` (used for mixed or continued paragraphs).

### 3.3 Bold Body — Style `a2` (متن ضخيم)

Inherits `a`, adds `bold`. Font: Zar + Times New Roman, same 14 pt.

### 3.4 Chapter / Section Heading — Style `Heading11` (فصل‌ها)

| Property | Value |
|---|---|
| styleId | `Heading11` |
| Justification | right |
| Font size | **14 pt** (28 half-pts) |
| Bold | Yes |
| bidi | false (overridden per paragraph as needed) |

### 3.5 TOC Styles

| Style | Font (cs) | Size | Bold | Indent |
|---|---|---|---|---|
| TOC1 | (default) | 10 pt | Yes | 0 |
| TOC2 | Times New Roman | 10 pt | No | 240 twips (left) |
| TOC3–TOC9 | Times New Roman | 10 pt | No | increasing |

### 3.6 Cover / Title Styles

| Style | CS Font | CS Size | Bold | Justification |
|---|---|---|---|---|
| Title1 | Nazanin | **18 pt** | Yes | center |
| Title2 | (inherits) | 16 pt | Yes | both (justified) |
| `a0` (عنوان فهرست) | — | 16 pt (cs 32 half-pts) | Yes | center |

### 3.7 Figure / Table Caption — Style `-` (شکل - جدول)

| Property | Value |
|---|---|
| Justification | center |
| ASCII size | 9 pt |
| Complex-script size | 10 pt |
| Line spacing | 240/auto (single) |
| bidi | false |

### 3.8 Page Number — Style `PageNumber`

| Property | Value |
|---|---|
| ASCII font | Times New Roman |
| CS font | **Zar** |
| ASCII size | 11 pt |
| CS size | 13 pt |

---

## 4. Document Direction & Bidi

- **Global section direction:** `<w:bidi/>` present → **RTL throughout**
- **RTL gutter:** `<w:rtlGutter/>` → binding margin is on the **right**
- **Theme language:** `bidi = ar-SA` (Arabic locale used as proxy; Persian inherits)
- **Paragraph bidi:** set explicitly on Persian paragraphs (`<w:bidi/>` in `<w:pPr>`)
- **Run RTL:** individual runs use `<w:rtl/>` where direction switches
- **Persian justification:** `lowKashida` (not `both`) — Word stretches kashida characters before expanding spaces

---

## 5. Page Numbering

| Section | Number format | Notes |
|---|---|---|
| Front matter (cover, abstract, toc) | `arabicAbjad` | Persian abjad numerals (ا، ب، ج…) |
| Main body chapters | Decimal (implicit from `start=1`) | Standard Arabic-Indic digits in Persian style |
| Footer content | `\u200cب`, `\u200cي`, etc. | Right-aligned footer with page field |

Page numbers appear in the **footer**, right-aligned, in Zar 13 pt.

---

## 6. Numbering (Lists)

| Abstract | Level 0 format | Nesting pattern |
|---|---|---|
| 0 | `%1-` decimal + dash | `%2.` `%3.` …  |
| 1 | chapter `4` (fixed) | `%1-%2`, `%1-%2.%3` … (chapter-section format) |
| 3 | `%1` decimal | `%1-%2`, `%1-%2.%3` … |
| 4 | `%1)` | lowerLetter, lowerRoman |
| 5 | bullet `•` (U+F0B7 Wingdings) | nested bullets |

**Section numbering pattern used in document:** `4-1`, `4-1.1`, `4-1.1.1` (chapter-section-subsection with dash separator).

---

## 7. Headers (Running Heads)

Headers are chapter-title running heads in the right margin area, RTL, in the custom body font. Each chapter section carries its own header text (e.g., "مروری بر مبانی نظری و مطالعات گذشته").

TOC sections have their own headers: "فهرست مطالب", "فهرست اشکال", etc.

---

## 8. Summary for LaTeX/XeLaTeX Implementation

```latex
% Required packages
\usepackage{polyglossia}
\setmainlanguage{persian}
\setotherlanguage{english}
\usepackage{fontspec}

% Fonts
\setmainfont{Zar}[Script=Arabic]           % body Persian
\newfontfamily\headingfont{B Nazanin}[Script=Arabic]  % headings
\newfontfamily\titlefont{Nazanin}[Script=Arabic]      % cover title
\setlatinfont{Times New Roman}              % Latin inside text

% Page geometry (RTL — right is binding side)
\usepackage[
  a4paper,
  top=3cm, bottom=3cm,
  inner=4cm,   % binding = right in RTL → use inner
  outer=3cm
]{geometry}

% Font sizes
% Body: 14pt cs (use \fontsize{14}{16.8}\selectfont or set as \normalsize)
% Chapter headings: 14pt bold
% Cover title: 18pt bold
% Captions: 10pt
% Page numbers: 13pt Zar
% TOC entries: 10pt

% Line spacing (body style 'a')
\usepackage{setspace}
\setstretch{1.2}   % approximately 288/240 = 1.2× single spacing

% Paragraph spacing
\setlength{\parskip}{0pt}
\setlength{\parindent}{...}  % check from indent values
```

---

## 9. Files in the Template

| File | Purpose |
|---|---|
| `word/styles.xml` | All named styles |
| `word/document.xml` | Body content + section properties |
| `word/numbering.xml` | List/numbering definitions |
| `word/theme/theme1.xml` | Office theme (Aptos fonts, color scheme) |
| `word/header[1-34].xml` | Chapter running headers (RTL) |
| `word/footer[1-15].xml` | Page number footers |
| `word/fontTable.xml` | Embedded font declarations |
| `word/media/` | Images (university logo, signature scans) |
