import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from PIL import Image
import numpy as np
from datetime import datetime
import io
import os

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, Image as RLImage
)


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(page_title="Autonomous Infrastructure Inspection")

st.title("Autonomous Infrastructure Inspection System")


# =========================
# ABOUT PROJECT SECTION
# =========================

st.subheader("About This Project")

st.write("""
This AI-based Autonomous Infrastructure Inspection System detects road surface damage
automatically using deep learning models based on YOLOv8.

The system can detect:

• Road Cracks (Segmentation Model)
• Potholes (Object Detection Model)
• Rust / Surface Corrosion Areas

Purpose of this system:

To assist infrastructure monitoring departments and smart city systems
in identifying damaged road areas quickly and accurately using image
and video-based inspection instead of manual surveys.
""")


# =========================
# LOAD MODELS
# =========================

@st.cache_resource
def load_models():
    crack_model   = YOLO("models/crack_Seg.pt")
    pothole_model = YOLO("models/pothole_rust.pt")
    return crack_model, pothole_model

crack_model, pothole_model = load_models()


# =========================
# PDF REPORT GENERATION
# =========================

def generate_pdf_report(crack_detected, pothole_detected,
                         crack_count, pothole_count,
                         annotated_image_np, source_name):
    """
    Generate a professional PDF report with:
    - Summary table
    - Annotated result image embedded
    - Recommendations
    """

    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=letter,
                               leftMargin=0.75*inch, rightMargin=0.75*inch,
                               topMargin=0.75*inch,  bottomMargin=0.75*inch)

    styles = getSampleStyleSheet()
    story  = []

    # ── Title style ──
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=20,
        textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=6,
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#16213e"),
        spaceBefore=14,
        spaceAfter=6,
    )
    normal_style = ParagraphStyle(
        "CustomNormal",
        parent=styles["Normal"],
        fontSize=10,
        leading=16,
    )

    # ── Header ──
    story.append(Paragraph("Autonomous Infrastructure Inspection Report", title_style))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor("#e94560"), spaceAfter=10))

    # ── Meta info table ──
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_data = [
        ["Field",          "Details"],
        ["Report Date",    now],
        ["Source File",    source_name],
        ["Models Used",    "YOLOv8 Segmentation + YOLOv8 Detection"],
        ["System",         "Autonomous Infrastructure Inspection AI"],
    ]
    meta_table = Table(meta_data, colWidths=[2*inch, 4.5*inch])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("BACKGROUND",    (0, 1), (0, -1),  colors.HexColor("#f0f0f0")),
        ("FONTNAME",      (0, 1), (0, -1),  "Helvetica-Bold"),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
        ("PADDING",       (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 16))

    # ── Detection Summary ──
    story.append(Paragraph("Detection Summary", heading_style))

    def status_text(detected):
        return "DETECTED" if detected else "NOT DETECTED"

    def status_color(detected):
        return colors.HexColor("#c0392b") if detected else colors.HexColor("#27ae60")

    summary_data = [
        ["Defect Type",   "Status",              "Count"],
        ["Road Crack",    status_text(crack_detected),   str(crack_count)],
        ["Pothole",       status_text(pothole_detected), str(pothole_count)],
    ]
    summary_table = Table(summary_data, colWidths=[2.5*inch, 2.5*inch, 1.5*inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#16213e")),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 11),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING",     (0, 0), (-1, -1), 8),
        # Crack row color
        ("TEXTCOLOR",   (1, 1), (1, 1),
         colors.HexColor("#c0392b") if crack_detected else colors.HexColor("#27ae60")),
        ("FONTNAME",    (1, 1), (1, 1),   "Helvetica-Bold"),
        # Pothole row color
        ("TEXTCOLOR",   (1, 2), (1, 2),
         colors.HexColor("#c0392b") if pothole_detected else colors.HexColor("#27ae60")),
        ("FONTNAME",    (1, 2), (1, 2),   "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 16))

    # ── Annotated Result Image ──
    story.append(Paragraph("Detection Result Image", heading_style))
    story.append(Paragraph(
        "The image below shows the annotated output with all detected defects highlighted.",
        normal_style
    ))
    story.append(Spacer(1, 8))

    # Convert numpy BGR → RGB → PIL → save to buffer
    if annotated_image_np is not None:
        if len(annotated_image_np.shape) == 3 and annotated_image_np.shape[2] == 3:
            img_rgb = cv2.cvtColor(annotated_image_np, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = annotated_image_np

        pil_img  = Image.fromarray(img_rgb.astype(np.uint8))
        img_buf  = io.BytesIO()
        pil_img.save(img_buf, format="PNG")
        img_buf.seek(0)

        # Fit image within page width
        max_w = 6.5 * inch
        orig_w, orig_h = pil_img.size
        ratio   = max_w / orig_w
        img_h   = orig_h * ratio

        rl_img = RLImage(img_buf, width=max_w, height=img_h)
        story.append(rl_img)

    story.append(Spacer(1, 16))

    # ── Recommendations ──
    story.append(Paragraph("Recommendations", heading_style))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#cccccc"), spaceAfter=8))

    recs = []
    if crack_detected:
        recs.append("• Road cracks detected — schedule crack sealing/filling within 30 days.")
    if pothole_detected:
        recs.append("• Potholes detected — immediate patching required to prevent vehicle damage.")
    if not crack_detected and not pothole_detected:
        recs.append("• No critical defects found. Continue routine monitoring schedule.")

    recs.append("• Re-inspect the flagged area after repairs are completed.")
    recs.append("• Maintain inspection logs for infrastructure health tracking.")

    for rec in recs:
        story.append(Paragraph(rec, normal_style))

    story.append(Spacer(1, 20))

    # ── Footer ──
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#cccccc"), spaceAfter=6))
    story.append(Paragraph(
        f"Generated by Autonomous Infrastructure Inspection System &nbsp;|&nbsp; {now}",
        ParagraphStyle("Footer", parent=styles["Normal"],
                       fontSize=8, textColor=colors.grey, alignment=1)
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# =========================
# PROCESS IMAGE PIPELINE
# =========================

def process_image_pipeline(image):

    image_np = np.array(image)
    output   = image_np.copy()

    crack_result   = crack_model.predict(image_np,  conf=0.25)[0]
    pothole_result = pothole_model.predict(image_np, conf=0.25)[0]

    crack_detected   = False
    pothole_detected = False
    crack_count      = 0
    pothole_count    = 0

    y_position = 30

    # ── Crack ──
    if crack_result.masks is not None:
        output         = crack_result.plot(img=output)
        crack_detected = True
        crack_count    = len(crack_result.masks)
    else:
        cv2.putText(output, "No Crack",
                    (10, y_position), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    y_position += 25

    # ── Pothole ──
    if len(pothole_result.boxes) > 0:
        output           = pothole_result.plot(img=output)
        pothole_detected = True
        pothole_count    = len(pothole_result.boxes)
    else:
        cv2.putText(output, "No Pothole",
                    (10, y_position), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    return output, crack_detected, pothole_detected, crack_count, pothole_count


# =========================
# PROCESS VIDEO PIPELINE
# =========================

def process_video_pipeline(video_path):

    cap    = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    writer = cv2.VideoWriter(output_path,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))

    crack_detected   = False
    pothole_detected = False
    crack_count      = 0
    pothole_count    = 0
    last_frame       = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = frame.copy()

        crack_result   = crack_model.predict(frame,   conf=0.5)[0]
        pothole_result = pothole_model.predict(frame,  conf=0.25)[0]

        if crack_result.masks is not None:
            output         = crack_result.plot(img=output)
            crack_detected = True
            crack_count   += len(crack_result.masks)

        if len(pothole_result.boxes) > 0:
            output           = pothole_result.plot(img=output)
            pothole_detected = True
            pothole_count   += len(pothole_result.boxes)

        writer.write(output)
        last_frame = output

    cap.release()
    writer.release()

    return output_path, last_frame, crack_detected, pothole_detected, crack_count, pothole_count


# =========================
# FILE UPLOADER
# =========================

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi"]
)


# =========================
# PROCESS BUTTON
# =========================

if uploaded_file is not None:

    file_type = uploaded_file.type

    if st.button("Process Detection"):

        # ════════════════════════
        # IMAGE
        # ════════════════════════
        if "image" in file_type:

            image = Image.open(uploaded_file)

            st.image(image, caption="Original Image", use_container_width=True)

            with st.spinner("Running detection…"):
                result, crack_det, pothole_det, crack_cnt, pothole_cnt = \
                    process_image_pipeline(image)

            # ── Result image ──
            st.subheader("Detection Result")
            st.image(result, caption="Annotated Detection Output",
                     use_container_width=True)

            # ── Explainability panel ──
            st.subheader("Detection Explainability")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Crack Detection**")
                if crack_det:
                    st.error(f"Crack Detected — {crack_cnt} area(s) found")
                    # Show only crack overlay
                    crack_only = np.array(image)
                    cr = crack_model.predict(crack_only, conf=0.25)[0]
                    if cr.masks is not None:
                        st.image(cr.plot(), caption="Crack Segmentation Only",
                                 use_container_width=True)
                else:
                    st.success("No Crack Detected")

            with col2:
                st.markdown("**Pothole Detection**")
                if pothole_det:
                    st.error(f"Pothole Detected — {pothole_cnt} pothole(s) found")
                    # Show only pothole overlay
                    pothole_only = np.array(image)
                    pr = pothole_model.predict(pothole_only, conf=0.25)[0]
                    if len(pr.boxes) > 0:
                        st.image(pr.plot(), caption="Pothole Detection Only",
                                 use_container_width=True)
                else:
                    st.success("No Pothole Detected")

            # ── PDF Report ──
            st.subheader("Inspection Report")
            pdf_bytes = generate_pdf_report(
                crack_det, pothole_det,
                crack_cnt, pothole_cnt,
                result, uploaded_file.name
            )

            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="inspection_report.pdf",
                mime="application/pdf"
            )

        # ════════════════════════
        # VIDEO
        # ════════════════════════
        elif "video" in file_type:

            temp_video = tempfile.NamedTemporaryFile(delete=False)
            temp_video.write(uploaded_file.read())

            st.video(temp_video.name)

            with st.spinner("Processing video… this may take a while"):
                out_video, last_frame, crack_det, pothole_det, crack_cnt, pothole_cnt = \
                    process_video_pipeline(temp_video.name)

            # ── Annotated video ──
            st.subheader("Detection Result Video")
            st.video(out_video)

            # ── Explainability — show last annotated frame ──
            st.subheader("Detection Explainability (Last Frame)")
            if last_frame is not None:
                st.image(last_frame,
                         caption="Last Annotated Frame — All Detections",
                         use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                if crack_det:
                    st.error(f"Crack Detected — {crack_cnt} instance(s) across video")
                else:
                    st.success("No Crack Detected in Video")
            with col2:
                if pothole_det:
                    st.error(f"Pothole Detected — {pothole_cnt} instance(s) across video")
                else:
                    st.success("No Pothole Detected in Video")

            # ── PDF Report ──
            st.subheader("Inspection Report")
            pdf_bytes = generate_pdf_report(
                crack_det, pothole_det,
                crack_cnt, pothole_cnt,
                last_frame, uploaded_file.name
            )

            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="inspection_report.pdf",
                mime="application/pdf"
            )

            st.success("Processing Completed Successfully")