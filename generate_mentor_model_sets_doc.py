"""Script to generate a professional Word document (.docx) and Markdown file (.md)
detailing the 4 Model Sets, Architectures, and Training Plan for the Mentor.
"""
import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

DOC_PATH = r"d:\code\Mangoleaf\Model_Sets_and_Training_Plan.docx"
MD_PATH = r"d:\code\Mangoleaf\Model_Sets_and_Training_Plan.md"

def set_cell_background(cell, fill_hex):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def create_document():
    doc = docx.Document()
    
    # Page Margins
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)
        
    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    
    # Document Header
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    run_title = title_p.add_run("RESEARCH PROJECT SPECIFICATION REPORT")
    run_title.font.size = Pt(18)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D) # Navy
    
    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(14)
    run_sub = sub_p.add_run("Multi-Set Vision Model Architecture, Distillation Framework, and Experimental Roadmap\nTarget Journal: Smart Agricultural Technology (Elsevier)")
    run_sub.font.size = Pt(11)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    
    # Metadata Block
    meta_table = doc.add_table(rows=2, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Project:", "Automated Mango Leaf Pathology via Trustworthy Distillation"),
        ("Dataset:", "8 Classes, Group-Aware Leak-Free Split (dHash Filtered)"),
        ("Compute Infrastructure:", "NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)"),
        ("Target Output:", "10-Model Zoo across 4 Paired Teacher-Student Sets")
    ]
    for idx, (label, val) in enumerate(meta_data):
        row = idx // 2
        col = idx % 2
        cell = meta_table.cell(row, col)
        set_cell_background(cell, "F2F5F9")
        set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(2)
        r1 = p.add_run(f"{label} ")
        r1.bold = True
        r1.font.size = Pt(9.5)
        r2 = p.add_run(val)
        r2.font.size = Pt(9.5)
        
    doc.add_paragraph().paragraph_format.space_after = Pt(10)
    
    # Section 1: Executive Overview
    h1 = doc.add_paragraph()
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(6)
    r_h1 = h1.add_run("1. Executive Overview & Methodology Design")
    r_h1.font.size = Pt(14)
    r_h1.font.bold = True
    r_h1.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    
    p_exec = doc.add_paragraph()
    p_exec.paragraph_format.space_after = Pt(8)
    p_exec.add_run(
        "To ensure high empirical rigor for Elsevier submission, the study evaluates automated disease detection "
        "not simply through standalone accuracy, but through a structured 4-Model System implemented across 4 distinct "
        "architectural families. Each family pairs one heavy Teacher model with three lightweight Student variants to isolate "
        "the precise contribution of knowledge distillation (KD) and explanation alignment (XKD)."
    )
    
    # Callout Box: 4-Model Paradigm
    callout = doc.add_table(rows=1, cols=1)
    callout.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_cell = callout.cell(0, 0)
    set_cell_background(c_cell, "EBF3FA")
    set_cell_margins(c_cell, top=100, bottom=100, left=150, right=150)
    cp = c_cell.paragraphs[0]
    cp.add_run("The Standard 4-Model Evaluation Quadrant (Applied to each Family):\n").bold = True
    cp.add_run("1. Teacher (Master Model): Heavy high-capacity network acting as domain expert.\n")
    cp.add_run("2. Student Plain (Baseline): Compact edge architecture trained only with hard cross-entropy.\n")
    cp.add_run("3. Student Distilled (Standard KD): Compact architecture trained with soft probability distillation (Hinton et al.).\n")
    cp.add_run("4. Student Explanation-Guided (XKD): Compact architecture trained with joint logit distillation and differentiable Grad-CAM attention alignment.")
    
    doc.add_paragraph().paragraph_format.space_after = Pt(10)
    
    # Section 2: The 4 Model Sets
    h2 = doc.add_paragraph()
    h2.paragraph_format.space_before = Pt(14)
    h2.paragraph_format.space_after = Pt(6)
    r_h2 = h2.add_run("2. The 4 Multi-Architectural Sets")
    r_h2.font.size = Pt(14)
    r_h2.font.bold = True
    r_h2.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    
    sets_info = [
        {
            "title": "SET 1: Hierarchical Vision Transformer Family (Primary Manuscript Focus)",
            "teacher": "Swin Transformer Tiny (Swin-T, Microsoft) — 27.5M params",
            "student": "MobileNetV3-Small (Google) — 1.5M params",
            "justification": "Contrasts a heavy windowed self-attention Transformer against an ultra-compact mobile inverted-residual CNN. Serves as the primary anchor for our 10 scientific experiments, XAI heatmap grid, and weather robustness stress-testing."
        },
        {
            "title": "SET 2: Modernized Pure-CNN Family (2024 State-of-the-Art)",
            "teacher": "ConvNeXt-Tiny (Meta AI) — 28.6M params",
            "student": "MobileNetV4-Small (Google 2024) — 3.8M params",
            "justification": "Tests whether modern pure-convolutional networks (7x7 depthwise filters) transfer reasoning better to newest-generation mobile DSP/NPU architectures than Transformers."
        },
        {
            "title": "SET 3: Compound Scaling & Agronomic Efficiency Family",
            "teacher": "EfficientNetV2-Small (Google) — 21.5M params",
            "student": "EfficientNet-B0 (Google) — 5.3M params",
            "justification": "Evaluates knowledge preservation within the universally recognized EfficientNet lineage, providing direct comparison with established agricultural AI benchmarks."
        },
        {
            "title": "SET 4: Multi-Axis Hybrid & Ultra-Edge Family",
            "teacher": "MaxViT-Tiny (Google Multi-Axis Attention) — 30.9M params",
            "student": "GhostNet (Huawei) — 3.9M params / MobileViT-XS (Apple) — 2.3M params",
            "justification": "Explores distillation from a hybrid convolution-attention teacher into ultra-low-compute linear operation models targeted for farm drones and low-battery field sensors."
        }
    ]
    
    for s in sets_info:
        sp = doc.add_paragraph()
        sp.paragraph_format.space_before = Pt(8)
        sp.paragraph_format.space_after = Pt(2)
        r_st = sp.add_run(s["title"])
        r_st.bold = True
        r_st.font.size = Pt(11.5)
        r_st.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
        
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_after = Pt(2)
        r_t = bp.add_run("Teacher Architecture: ")
        r_t.bold = True
        bp.add_run(s["teacher"])
        
        bp2 = doc.add_paragraph(style='List Bullet')
        bp2.paragraph_format.space_after = Pt(2)
        r_s = bp2.add_run("Student Architecture: ")
        r_s.bold = True
        bp2.add_run(s["student"])
        
        bp3 = doc.add_paragraph(style='List Bullet')
        bp3.paragraph_format.space_after = Pt(6)
        r_j = bp3.add_run("Scientific Rationale: ")
        r_j.bold = True
        bp3.add_run(s["justification"])
        
    doc.add_paragraph().paragraph_format.space_after = Pt(10)
    
    # Section 3: Master Model Inventory Table
    h3 = doc.add_paragraph()
    h3.paragraph_format.space_before = Pt(14)
    h3.paragraph_format.space_after = Pt(6)
    r_h3 = h3.add_run("3. Master Model Inventory & Specifications")
    r_h3.font.size = Pt(14)
    r_h3.font.bold = True
    r_h3.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    
    table = doc.add_table(rows=1, cols=6)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Set #", "Role", "Model Name", "Architecture Type", "Params", "Target Device"]
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        set_cell_background(hdr_cells[i], "1B365D")
        set_cell_margins(hdr_cells[i], top=100, bottom=100, left=100, right=100)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            r.font.size = Pt(9.5)
            
    rows_data = [
        ("Set 1", "Teacher", "Swin-T", "Hierarchical ViT", "27.5M", "Workstation / Cloud GPU"),
        ("Set 1", "Student", "MobileNetV3-Small", "Mobile Inverted CNN", "1.5M", "Budget Android Smartphone"),
        ("Set 2", "Teacher", "ConvNeXt-Tiny", "Modernized Pure-CNN", "28.6M", "Workstation / Cloud GPU"),
        ("Set 2", "Student", "MobileNetV4-Small", "Universal Edge CNN (2024)", "3.8M", "Modern Edge NPU / Smartphone"),
        ("Set 3", "Teacher", "EfficientNetV2-Small", "Progressive Fused-MBConv", "21.5M", "Workstation / Cloud GPU"),
        ("Set 3", "Student", "EfficientNet-B0", "Compound Scaled CNN", "5.3M", "Mid-Range Mobile Device"),
        ("Set 4", "Teacher", "MaxViT-Tiny", "Hybrid CNN-Attention", "30.9M", "Workstation / Cloud GPU"),
        ("Set 4", "Student", "GhostNet", "Cheap Linear Feature CNN", "3.9M", "IoT Sensor / Agricultural Drone"),
        ("Set 4", "Student", "MobileViT-XS", "Mobile Hybrid Transformer", "2.3M", "Low-Power Edge Device")
    ]
    
    for set_num, role, name, arch, params, dev in rows_data:
        row_cells = table.add_row().cells
        bg = "FFFFFF" if "Teacher" in role else "F8FAFC"
        for i, val in enumerate([set_num, role, name, arch, params, dev]):
            row_cells[i].text = val
            set_cell_background(row_cells[i], bg)
            set_cell_margins(row_cells[i], top=70, bottom=70, left=90, right=90)
            p = row_cells[i].paragraphs[0]
            if i in [0, 1, 4]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.size = Pt(9)
                if i == 2:
                    r.font.bold = True
                    
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # Section 4: Automated Evaluation Deliverables
    h4 = doc.add_paragraph()
    h4.paragraph_format.space_before = Pt(14)
    h4.paragraph_format.space_after = Pt(6)
    r_h4 = h4.add_run("4. Automated Evaluation Deliverables (Per Model)")
    r_h4.font.size = Pt(14)
    r_h4.font.bold = True
    r_h4.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    
    p_deliv = doc.add_paragraph()
    p_deliv.paragraph_format.space_after = Pt(6)
    p_deliv.add_run("Upon completion of training, the automated pipeline automatically computes and renders:")
    
    deliverables = [
        ("Confusion Matrix (300 DPI)", "Normalized and raw-count 8x8 heatmaps with per-class classification accuracy."),
        ("Multi-Class ROC & AUC Curves (300 DPI)", "One-vs-Rest ROC curve for each disease class plus macro-average Area Under Curve."),
        ("Training & Validation Dynamics (300 DPI)", "Epoch-by-epoch loss convergence, validation accuracy, and macro F1-score tracking."),
        ("Explainable AI (XAI) Alignment", "Grad-CAM and LIME heatmap similarity metrics (SSIM & Spearman rank correlation) against the Teacher."),
        ("Environmental Robustness Benchmarks", "Accuracy retention under simulated weather conditions (fog, rain, camera motion blur, low light).")
    ]
    for title, desc in deliverables:
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(3)
        r_bt = p.add_run(f"{title}: ")
        r_bt.bold = True
        p.add_run(desc)
        
    doc.save(DOC_PATH)
    print(f"Word Document saved -> {DOC_PATH}")

def create_markdown():
    md_content = """# Research Project Specification Report
## Multi-Set Vision Model Architecture, Distillation Framework, and Experimental Roadmap
**Target Journal**: *Smart Agricultural Technology* (Elsevier)

---

### 1. Executive Overview & Methodology Design
To ensure high empirical rigor for Elsevier submission, the study evaluates automated disease detection through a structured **4-Model System** implemented across **4 distinct architectural families**. Each family pairs one heavy Teacher model with three lightweight Student variants:

```
                  ┌──────────────────────────────────────────────┐
                  │                 1. TEACHER                   │
                  │             (Heavy Master Model)             │
                  └──────────────────────┬───────────────────────┘
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         │                               │                               │
         ▼                               ▼                               ▼
  2. STUDENT PLAIN              3. STUDENT DISTILLED               4. STUDENT XKD
  (Baseline: Cross-Entropy)     (Standard KD: Soft Logits)        (Explanation-Guided: CAM)
```

---

### 2. The 4 Multi-Architectural Sets

#### **SET 1: Hierarchical Vision Transformer Family (Primary Manuscript Focus)**
* **Teacher Architecture**: Swin Transformer Tiny (`swin_t`, Microsoft) — **27.5M parameters**
* **Student Architecture**: MobileNetV3-Small (`mnv3_s`, Google) — **1.5M parameters**
* **Scientific Rationale**: Contrasts a heavy windowed self-attention Transformer against an ultra-compact mobile inverted-residual CNN. Serves as the primary anchor for the 10 scientific experiments, XAI heatmap comparison grid, and environmental robustness testing.

#### **SET 2: Modernized Pure-CNN Family (2024 State-of-the-Art)**
* **Teacher Architecture**: ConvNeXt-Tiny (`convnext_t`, Meta AI) — **28.6M parameters**
* **Student Architecture**: MobileNetV4-Small (`mnv4_s`, Google 2024) — **3.8M parameters**
* **Scientific Rationale**: Tests whether modern pure-convolutional networks (7x7 depthwise kernels) transfer diagnostic reasoning better to newest-generation mobile edge NPU architectures than Transformers.

#### **SET 3: Compound Scaling & Agronomic Efficiency Family**
* **Teacher Architecture**: EfficientNetV2-Small (`effnetv2_s`, Google) — **21.5M parameters**
* **Student Architecture**: EfficientNet-B0 (`effnet_b0`, Google) — **5.3M parameters**
* **Scientific Rationale**: Evaluates knowledge preservation within the universally recognized EfficientNet lineage, providing direct parity with standard agricultural computer vision benchmarks.

#### **SET 4: Multi-Axis Hybrid & Ultra-Edge Family**
* **Teacher Architecture**: MaxViT-Tiny (`maxvit_t`, Google) — **30.9M parameters**
* **Student Architecture**: GhostNet (`ghostnet`, Huawei) — **3.9M parameters** / MobileViT-XS (`mobilevit_xs`, Apple) — **2.3M parameters**
* **Scientific Rationale**: Explores distillation from a hybrid convolution-attention teacher into ultra-low-compute linear operation models targeted for farm drones and low-battery IoT sensors.

---

### 3. Master Model Inventory Table

| Set # | Role | Model Name | Architecture Type | Parameters | Target Deployment Device |
| :---: | :---: | :--- | :--- | :---: | :--- |
| **Set 1** | Teacher | **Swin-T** | Hierarchical Vision Transformer | 27.5M | Workstation / Cloud GPU |
| **Set 1** | Student | **MobileNetV3-Small** | Mobile Inverted Residual CNN | 1.5M | Budget Android Smartphone |
| **Set 2** | Teacher | **ConvNeXt-Tiny** | Modernized Pure-CNN | 28.6M | Workstation / Cloud GPU |
| **Set 2** | Student | **MobileNetV4-Small** | Universal Edge CNN (2024) | 3.8M | Modern Smartphone Edge NPU |
| **Set 3** | Teacher | **EfficientNetV2-Small** | Progressive Fused-MBConv | 21.5M | Workstation / Cloud GPU |
| **Set 3** | Student | **EfficientNet-B0** | Compound Scaled CNN | 5.3M | Mid-Range Mobile Device |
| **Set 4** | Teacher | **MaxViT-Tiny** | Hybrid CNN + Self-Attention | 30.9M | Workstation / Cloud GPU |
| **Set 4** | Student | **GhostNet** | Cheap Linear Feature CNN | 3.9M | Farm Drone / IoT Sensor |
| **Set 4** | Student | **MobileViT-XS** | Mobile Hybrid Transformer | 2.3M | Low-Power Field Sensor |

---

### 4. Automated Evaluation Deliverables (Per Model)
1. **Confusion Matrix (300 DPI)**: Normalized and raw 8x8 heatmaps across all 8 mango leaf disease classes.
2. **Multi-Class ROC & AUC Curves (300 DPI)**: One-vs-Rest ROC curve for every disease class + Macro-Average Area Under Curve.
3. **Training & Validation Dynamics (300 DPI)**: Epoch-by-epoch loss convergence, validation accuracy, and macro F1-score tracking.
4. **Explainable AI (XAI) Lesion Fidelity**: Grad-CAM and LIME heatmap alignment (SSIM & Spearman correlation) against the Teacher.
5. **Environmental Robustness Stress-Testing**: Accuracy retention under simulated fog, rain, camera motion blur, and low lighting.
"""
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Markdown file saved -> {MD_PATH}")

if __name__ == "__main__":
    create_document()
    create_markdown()
