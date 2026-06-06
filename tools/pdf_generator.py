"""
PDF 报告生成工具
"""
import os
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)
from loguru import logger
from langchain_core.tools import tool


@tool
def generate_pdf(content: str, title: str = "Report", filename: str = None) -> str:
    """
    生成 PDF 报告文件

    Args:
        content: 报告内容（支持简单的 Markdown 格式）
        title: 报告标题
        filename: 文件名（可选，自动生成时使用时间戳）

    Returns:
        PDF 文件路径
    """
    logger.info(f"生成 PDF 报告: {title}")

    try:
        # 确保输出目录存在
        output_dir = Path("./output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        if filename is None:
            import time

            timestamp = int(time.time())
            filename = f"report_{timestamp}.pdf"

        if not filename.endswith(".pdf"):
            filename += ".pdf"

        file_path = output_dir / filename

        # 创建 PDF 文档
        doc = SimpleDocTemplate(
            str(file_path),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )

        # 样式设置
        styles = getSampleStyleSheet()

        # 自定义标题样式
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor("#2c3e50"),
            alignment=1,  # 居中
        )

        # 自定义正文样式
        body_style = ParagraphStyle(
            "CustomBody",
            parent=styles["Normal"],
            fontSize=12,
            leading=18,
            spaceAfter=12,
            textColor=colors.HexColor("#34495e"),
        )

        # 构建文档内容
        story = []

        # 标题
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.5 * inch))

        # 处理内容（简单的分段）
        sections = content.split("\n\n")
        for section in sections:
            section = section.strip()
            if not section:
                continue

            # 检测是否是标题（以 # 开头）
            if section.startswith("# "):
                heading_text = section[2:]
                story.append(Paragraph(heading_text, styles["Heading2"]))
            elif section.startswith("## "):
                heading_text = section[3:]
                story.append(Paragraph(heading_text, styles["Heading3"]))
            elif section.startswith("### "):
                heading_text = section[4:]
                story.append(Paragraph(heading_text, styles["Heading4"]))
            elif "|" in section and "-" in section[:50]:
                # 简单的表格处理
                lines = [line.strip() for line in section.split("\n") if line.strip()]
                table_data = []
                for line in lines:
                    if "---" not in line:
                        cells = [cell.strip() for cell in line.split("|")]
                        cells = [c for c in cells if c]  # 移除空单元格
                        if cells:
                            table_data.append(cells)

                if len(table_data) > 1:
                    # 创建表格
                    col_widths = [inch * 1.5] * len(table_data[0])
                    table = Table(table_data, colWidths=col_widths)
                    table.setStyle(
                        TableStyle(
                            [
                                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                ("FONTSIZE", (0, 0), (-1, 0), 10),
                                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
                                ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#dee2e6")),
                            ]
                        )
                    )
                    story.append(table)
                    story.append(Spacer(1, 0.2 * inch))
            else:
                # 普通段落
                paragraph = Paragraph(section.replace("\n", "<br/>"), body_style)
                story.append(paragraph)

        # 生成 PDF
        doc.build(story)

        return f"PDF 生成成功: {file_path.absolute()}"

    except Exception as e:
        logger.error(f"PDF 生成失败: {e}")
        return f"PDF 生成失败: {str(e)}"
