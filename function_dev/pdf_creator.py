from fpdf import FPDF
from datetime import datetime
import os


def export_json_to_pdf(data, output_path="summary_report.pdf", title="Summary Report"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    font_regular = os.path.join("fonts", "NotoSansKR-Regular.ttf")
    font_bold = os.path.join("fonts", "NotoSansKR-Bold.ttf")

    pdf.add_font("NotoSans", "", font_regular)
    pdf.add_font("NotoSans", "B", font_bold)

    width = pdf.w - 30  # 페이지 폭 - 여백 (15씩)
    line_height = 8

    # 타이틀 영역
    pdf.set_font("NotoSans", size=16)
    pdf.multi_cell(width, line_height, title)
    pdf.set_font("NotoSans", size=10)
    pdf.multi_cell(width, line_height, f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    pdf.ln(10)

    # 데이터 루프
    for idx, item in enumerate(data, 1):
        # 제목
        pdf.set_font("NotoSans", style="B", size=12)
        title_text = f"{idx}. 제목: {item.get('title', 'No Title')}"
        pdf.multi_cell(width, line_height, title_text)
        pdf.ln(2)

        # URL
        pdf.set_font("NotoSans", size=10)
        url_text = f"URL: {item.get('url', 'No URL')}"
        pdf.multi_cell(width, line_height, url_text)
        pdf.ln(2)

        # 요약
        pdf.set_font("NotoSans", size=11)
        summary_text = f"요약: {item.get('summary', 'No summary available')}"
        pdf.multi_cell(width, line_height, summary_text)
        pdf.ln(10)  # 항목 간 여백

    pdf.output(output_path)
    print(f"✅ PDF 생성 완료: {output_path}")
    return output_path

# 테스트 구동
if __name__ == "__main__":
    test_data = [
        {
            "title": "AI 뉴스 요약",
            "url": "https://example.com/ai-news",
            "summary": "이 뉴스는 AI 기술의 최신 동향에 대해 다룹니다. 인공지능은 앞으로 더욱 발전할 것으로 기대되며..."
        },
        {
            "title": "논문: 딥러닝 혁신",
            "url": "https://arxiv.org/abs/1234.5678",
            "summary": "본 논문은 딥러닝의 새로운 구조에 대해 설명하고 성능을 평가합니다. 실험 결과로는 기존 모델 대비 월등한 성능 향상을 보였습니다."
        }
    ]
    export_json_to_pdf(test_data, output_path="output.pdf")
